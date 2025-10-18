/* test-simple-text-buffer.c
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

#include "foundry-simple-text-buffer.h"

static void
assert_bytes (GBytes     *bytes,
              const char *expected)
{
  gsize len = 0;
  const char *str = g_bytes_get_data (bytes, &len);

  g_assert_cmpint (len, ==, strlen (expected));

  if (len > 0)
    g_assert_cmpstr (str, ==, expected);
}

static void
test_simple_text_buffer (void)
{
  g_autoptr(FoundryTextBuffer) buffer = foundry_simple_text_buffer_new ();

  {
    g_autoptr(GBytes) contents = foundry_text_buffer_dup_contents (buffer);

    assert_bytes (contents, "");
  }

  {
    g_autoptr(FoundryTextEdit) first = foundry_text_edit_new (NULL, 0, 0, 0, 0, "first\n");
    g_autoptr(FoundryTextEdit) second = foundry_text_edit_new (NULL, 0, -1, 1, 0, "\nsecond\n");
    g_autoptr(FoundryTextEdit) third = foundry_text_edit_new (NULL, 1, 0, 1, 0, "third\n");
    g_autoptr(GBytes) contents = NULL;

    g_assert_true (foundry_text_buffer_apply_edit (buffer, first));
    g_assert_true (foundry_text_buffer_apply_edit (buffer, third));
    g_assert_true (foundry_text_buffer_apply_edit (buffer, second));

    contents = foundry_text_buffer_dup_contents (buffer);

    assert_bytes (contents, "first\nsecond\nthird\n");
  }
}

int
main (int argc,
      char *argv[])
{
  g_test_init (&argc, &argv, NULL);
  g_test_add_func ("/Foundry/SimpleTextBuffer/basic", test_simple_text_buffer);
  return g_test_run ();
}
