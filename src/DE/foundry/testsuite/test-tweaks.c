/* test-tweaks.c
 *
 * Copyright 2024 Christian Hergert <chergert@redhat.com>
 *
 * This library is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2.1 of the
 * License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#include "config.h"

#include <foundry.h>

#include "foundry-tweak-path.h"

#include "test-util.h"

static void
test_tweak_path_fiber (void)
{
  g_autoptr(FoundryTweakPath) root = foundry_tweak_path_new ("/");
  g_autoptr(FoundryTweakPath) app = foundry_tweak_path_new ("/app");
  g_autoptr(FoundryTweakPath) user = foundry_tweak_path_new ("/user");

  g_assert_true (foundry_tweak_path_has_prefix (app, root));
  g_assert_true (foundry_tweak_path_has_prefix (user, root));
  g_assert_false (foundry_tweak_path_equal (root, app));
  g_assert_false (foundry_tweak_path_equal (user, app));
  g_assert_false (foundry_tweak_path_has_prefix (root, app));
  g_assert_false (foundry_tweak_path_has_prefix (root, user));

  {
    g_autoptr(FoundryTweakPath) basic = foundry_tweak_path_new ("/app/basic");
    g_assert_true (foundry_tweak_path_has_prefix (basic, app));
    g_assert_false (foundry_tweak_path_equal (basic, app));
    g_assert_false (foundry_tweak_path_has_prefix (app, basic));

    {
      g_autoptr(FoundryTweakPath) basic2 = foundry_tweak_path_push (app, "basic");
      g_assert_true (foundry_tweak_path_has_prefix (basic2, app));
      g_assert_false (foundry_tweak_path_has_prefix (basic2, basic));
      g_assert_true (foundry_tweak_path_equal (basic2, basic));

      {
        g_autoptr(FoundryTweakPath) app2 = foundry_tweak_path_pop (basic2);
        g_assert_nonnull (app2);
        g_assert_true (foundry_tweak_path_equal (app2, app));
        g_assert_false (foundry_tweak_path_equal (app2, basic2));
      }
    }

    {
      g_autoptr(FoundryTweakPath) basic2 = foundry_tweak_path_new ("/user/basic");
      g_assert_false (foundry_tweak_path_equal (basic2, basic));
      g_assert_false (foundry_tweak_path_has_prefix (basic2, app));
    }
  }

  {
    g_autoptr(FoundryTweakPath) a = foundry_tweak_path_new ("/app/basic/thing");
    g_autoptr(FoundryTweakPath) b = foundry_tweak_path_new ("/app/basic/thing/");

    g_assert_true (foundry_tweak_path_equal (a, b));
  }

  {
    g_autoptr(FoundryTweakPath) a = foundry_tweak_path_new ("/app/basic/thing");
    g_autoptr(FoundryTweakPath) b = foundry_tweak_path_push (app, "basic/thing/");

    g_assert_true (foundry_tweak_path_equal (a, b));
  }
}

static void
test_tweak_path (void)
{
  test_from_fiber (test_tweak_path_fiber);
}

int
main (int argc,
      char *argv[])
{
  dex_init ();
  g_test_init (&argc, &argv, NULL);
  g_test_add_func ("/Foundry/Tweak/Path/basic", test_tweak_path);
  return g_test_run ();
}
