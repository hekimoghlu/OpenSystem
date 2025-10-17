/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, February 5, 2024.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */

#include <cstdint>
#include <entt/entt.hpp>

struct position {
  float x;
  float y;
};

struct velocity {
  float dx;
  float dy;
};

void update(entt::registry &registry) {
  auto view = registry.view<position, velocity>();

  for (auto entity : view) {
    // gets only the components that are going to be used ...

    auto &vel = view.get<velocity>(entity);

    vel.dx = 0.;
    vel.dy = 0.;

    // ...
  }
}

void update(std::uint64_t dt, entt::registry &registry) {
  registry.view<position, velocity>().each([dt](auto &pos, auto &vel) {
    // gets all the components of the view at once ...

    pos.x += vel.dx * dt;
    pos.y += vel.dy * dt;

    // ...
  });
}

int main() {
  entt::registry registry;
  std::uint64_t dt = 16;

  for (auto i = 0; i < 10; ++i) {
    auto entity = registry.create();
    registry.assign<position>(entity, i * 1.f, i * 1.f);
    if (i % 2 == 0) {
      registry.assign<velocity>(entity, i * .1f, i * .1f);
    }
  }

  update(dt, registry);
  update(registry);

  // ...
}
