/* test-file.c
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

#include "test-util.h"

static void
test_find_with_depth_fiber (void)
{
  g_autoptr(GFile) srcdir = g_file_new_for_path (g_getenv ("G_TEST_SRCDIR"));
  g_autoptr(GFile) dir = g_file_get_child (srcdir, "test-file");
  g_autoptr(GError) error = NULL;
  GPtrArray *ar;

  ar = dex_await_boxed (foundry_file_find_with_depth (dir, "*.json", 1), &error);
  g_assert_no_error (error);
  g_assert_nonnull (ar);
  g_assert_cmpint (ar->len, ==, 0);
  g_clear_pointer (&ar, g_ptr_array_unref);

  ar = dex_await_boxed (foundry_file_find_with_depth (dir, "*.json", 2), &error);
  g_assert_no_error (error);
  g_assert_nonnull (ar);
  g_assert_cmpint (ar->len, ==, 1);
  g_clear_pointer (&ar, g_ptr_array_unref);

  ar = dex_await_boxed (foundry_file_find_with_depth (dir, "*.json", 3), &error);
  g_assert_no_error (error);
  g_assert_nonnull (ar);
  g_assert_cmpint (ar->len, ==, 2);
  g_clear_pointer (&ar, g_ptr_array_unref);

  ar = dex_await_boxed (foundry_file_find_with_depth (dir, "*.json", 4), &error);
  g_assert_no_error (error);
  g_assert_nonnull (ar);
  g_assert_cmpint (ar->len, ==, 3);
  g_clear_pointer (&ar, g_ptr_array_unref);

  ar = dex_await_boxed (foundry_file_find_with_depth (dir, "*.json", 5), &error);
  g_assert_no_error (error);
  g_assert_nonnull (ar);
  g_assert_cmpint (ar->len, ==, 3);
  g_clear_pointer (&ar, g_ptr_array_unref);
}

static void
test_find_with_depth (void)
{
  test_from_fiber (test_find_with_depth_fiber);
}

int
main (int argc,
      char *argv[])
{
  const char *srcdir = g_getenv ("G_TEST_SRCDIR");

  g_assert_nonnull (srcdir);

  dex_init ();

  g_test_init (&argc, &argv, NULL);
  g_test_add_func ("/Foundry/File/find_with_depth", test_find_with_depth);

  return g_test_run ();
}
