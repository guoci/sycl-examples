// SYCL implementation of N-body simulation for Maxwell–Boltzmann distribution approximation
#include <SFML/Graphics.hpp>
#include <sycl/sycl.hpp>
#include <thread>
#include <random>


constexpr auto sqrt_N = 128;
constexpr auto N = sqrt_N * sqrt_N; // number of particles
constexpr auto dt = std::chrono::duration<float>{0.000008f * .05}; // time step
constexpr auto init_velocity = 500.f; // initial velocity of particles
constexpr auto particle_radius = 1.f; // on-screen particle radius

int main(const int argc, const char *const *const argv) {
    std::vector<sycl::float2> positions(N);
    std::random_device rd;
    std::mt19937 gen{rd()};
    std::uniform_real_distribution<> position_distribution(0.f, 1.f);
    // random positions for particles
    if (false)
        std::ranges::generate(positions, [&gen, &position_distribution]() mutable {
            return sycl::float2{position_distribution(gen), position_distribution(gen)};
        });

    // evenly spaced particles
    for (size_t i = 0; i < sqrt_N; ++i)
        for (size_t j = 0; j < sqrt_N; ++j)
            positions[i * sqrt_N + j] = sycl::float2{i, j} / sqrt_N;

    std::vector<sycl::float2> velocities(N);
    std::uniform_real_distribution<> direction_distribution(0.f, 2 * std::numbers::pi_v<float>);

    // random directions for particles
    std::ranges::generate(velocities, [&gen, &direction_distribution]()mutable {
        const auto aa = direction_distribution(gen);
        return sycl::float2{init_velocity * std::sin(aa), init_velocity * std::cos(aa)};
    });

    sycl::queue q{sycl::gpu_selector_v};
    const unsigned particles_window_width = 1000;
    const unsigned window_height = 1000;
    const unsigned distribution_window_width = 1200;
    sf::RenderWindow window(sf::VideoMode(particles_window_width + distribution_window_width, window_height),
                            "N-body simulation for Maxwell-Boltzmann distribution approximation");
    sycl::buffer buf_pos{positions};
    sycl::buffer buf_vel{velocities};
    sycl::buffer<int, 1> buf_mod{sycl::range<1>{N}};

    const auto time_steps = 1024;
    for (int simulation_iter = 0; simulation_iter < time_steps; ++simulation_iter) {
        window.clear();
        const auto start = std::chrono::steady_clock::now();
        // particle to wall collisions
        q.fill(sycl::accessor{buf_mod, sycl::write_only, sycl::no_init}, 0);
        q.submit([&](sycl::handler &h) {
            sycl::accessor pos{buf_pos, h, sycl::read_only};
            sycl::accessor vel{buf_vel, h, sycl::read_write};
            const auto radius = particle_radius / particles_window_width / 2;
            h.parallel_for<class particle_wall_collisions>(sycl::range<1>{N}, [=](sycl::item<1> it) {
                auto v1 = vel[it];
                const auto p1 = pos[it];
                if (p1.x() > 1 - radius)
                    v1 = {-sycl::abs(v1.x()), v1.y()};
                if (p1.x() < 0 + radius)
                    v1 = {sycl::abs(v1.x()), v1.y()};
                if (p1.y() > 1 - radius)
                    v1 = {v1.x(), -sycl::abs(v1.y())};
                if (p1.y() < 0 + radius)
                    v1 = {v1.x(), sycl::abs(v1.y())};
                vel[it] = v1;
            });
        });

        // particle to particle collisions
        q.submit([&](sycl::handler &h) {
            sycl::accessor pos{buf_pos, h, sycl::read_only};
            sycl::accessor vel{buf_vel, h, sycl::write_only};
            sycl::accessor mod{buf_mod, h, sycl::read_write};
            h.parallel_for<class particle_particle_collisions>(sycl::range<1>{N - 1}, [=](sycl::item<1> it) {
                const auto p1 = pos[it];
                for (size_t i = it.get_id(0) + 1; i < N; ++i) {
                    const auto p2 = pos[i];
                    const auto distsq = sycl::dot(p1 - p2, p1 - p2);
                    const auto contact_dist = 2 * particle_radius / particles_window_width;
                    const auto collide = distsq < contact_dist * contact_dist;
                    if (collide) {
                        sycl::atomic_ref<int, sycl::memory_order::acq_rel, sycl::memory_scope::device, sycl::access::address_space::global_space>
                                self_mod{mod[it]}, other_mod{mod[i]};
                        // allow only one collision for self_mod particle
                        const auto s = self_mod.fetch_add(1);
                        const auto o = other_mod.fetch_add(1);
                        if (s == 0 && o == 0) {
                            const auto v1 = vel[it];
                            const auto v2 = vel[i];
                            vel[it] = v1 - sycl::dot(v1 - v2, p1 - p2) / distsq * (p1 - p2);
                            vel[i] = v2 - sycl::dot(v1 - v2, p1 - p2) / distsq * (p2 - p1);
                            break;
                        }else {
                            self_mod.fetch_sub(1);
                            other_mod.fetch_sub(1);
                            if (s != 0)
                                break;
                        }
                    }
                }
            });
        });
        // update particle positions
        q.submit([&](sycl::handler &h) {
            sycl::accessor pos{buf_pos, h, sycl::read_write};
            sycl::accessor vel{buf_vel, h, sycl::read_only};
            h.parallel_for(
                    sycl::range<1>{N},
                    [=](sycl::item<1> it) {
                        pos[it] = pos[it] + dt.count() * vel[it];
                    }
            );
        });

        // calculate the total energy
        sycl::buffer<float> sum_vel{1};
        q.submit([&](sycl::handler &cgh) {
            sycl::accessor vel{buf_vel, cgh, sycl::read_only};
            cgh.parallel_for(
                    sycl::range<1>{velocities.size()},
                    sycl::reduction(sum_vel, cgh, sycl::plus<>{}, sycl::property::reduction::initialize_to_identity{}),
                    [=](sycl::id<1> idx, auto &sum) {
                        sum.combine(sycl::dot(vel[idx], vel[idx]));
                    });
        });
        const auto sumv = sum_vel.get_host_access()[0];

        // print stats
        std::cout << "===== iteration: " << simulation_iter << '\n';
        std::cout << "mean of squared velocities: " << sumv / N << '\n';
        const auto diff = (std::chrono::steady_clock::now() - start);
        std::cout << "time taken: " << std::chrono::duration<double, std::milli>(diff).count() << "ms\n";


        // draw particles
        for (const auto &position: buf_pos.get_host_access()) {
            auto shape = sf::CircleShape(particle_radius);
            shape.setPosition(position.x() * 1000 - particle_radius, position.y() * 1000 - particle_radius);
            window.draw(shape);
        }

        // draw histogram of velocities
        const auto dist_height_scale = 400.f;
        std::vector<ssize_t> bins(333);
        const auto binswidth = 32.f;
        for (const auto &velocity: buf_vel.get_host_access()) {
            const auto bin_idx = static_cast<size_t>(sycl::length(velocity) / binswidth);
            if (bin_idx < bins.size())
                ++bins[bin_idx];
        }
        for (size_t idx = 0; idx < bins.size(); ++idx) {
            const auto hist_height = static_cast<float>(window_height) * dist_height_scale *
                                     static_cast<float>(bins[idx]) / N / binswidth;
            sf::RectangleShape rect{sf::Vector2f{binswidth, hist_height}};
            rect.setPosition(sf::Vector2f{particles_window_width + binswidth * static_cast<float>(idx),
                                          window_height - hist_height});
            window.draw(rect);
        }

        // draw theoretical distribution
        const auto a = 2 / init_velocity / init_velocity;
        const auto gridsize = 1000;
        sf::Vertex curve[gridsize];
        for (int thickness_idx = -2; thickness_idx <= 2; ++thickness_idx) {
            for (size_t grid_idx = 0; grid_idx < gridsize; ++grid_idx) {
                const auto x = static_cast<float>(grid_idx) / gridsize * init_velocity * 4;
                curve[grid_idx] = sf::Vertex(sf::Vector2f(particles_window_width + x,
                                                          window_height *
                                                          (1 - dist_height_scale * a * x * std::exp(-a * x * x / 2)) +
                                                          static_cast<float>(thickness_idx)));
                curve[grid_idx].color = sf::Color{255, 0, 0};
            }
            window.draw(curve, gridsize, sf::LineStrip);
        }

        // draw iteration count
        sf::Text iteration_count;
        const auto font = sf::Font{};
        iteration_count.setFont(font);
        iteration_count.setString(std::to_string(simulation_iter));
        iteration_count.setFillColor(sf::Color::White);
        iteration_count.setCharacterSize(32);
//        iteration_count.setPosition(500, 500);
        iteration_count.setStyle(sf::Text::Regular);
        window.draw(iteration_count);

        window.display();

        if (simulation_iter == 0)
            std::this_thread::sleep_for(std::chrono::seconds{4});
        if (simulation_iter + 1 == time_steps)
            std::this_thread::sleep_for(std::chrono::seconds{4});
    }
}
