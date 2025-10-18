/* foundry-dap-protocol.c
 *
 * Copyright 2025 Christian Hergert <chergert@redhat.com>
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

#include "foundry-dap-protocol.h"
#include "foundry-json-node.h"

gboolean
foundry_dap_protocol_has_error (JsonNode *node)
{
  gint64 request_seq = 0;
  gboolean success = FALSE;

  g_return_val_if_fail (node != NULL, FALSE);

  if (!FOUNDRY_JSON_OBJECT_PARSE (node, "type", "response"))
    return TRUE;

  if (!FOUNDRY_JSON_OBJECT_PARSE (node, "request_seq", FOUNDRY_JSON_NODE_GET_INT (&request_seq)))
    return TRUE;

  if (!FOUNDRY_JSON_OBJECT_PARSE (node, "success", FOUNDRY_JSON_NODE_GET_BOOLEAN (&success)) || !success)
    return TRUE;

  return FALSE;
}

GError *
foundry_dap_protocol_extract_error (JsonNode *node)
{
  const char *id = NULL;
  const char *format = NULL;

  g_return_val_if_fail (node != NULL, NULL);

  if (!FOUNDRY_JSON_OBJECT_PARSE (node, "error", "{",
                                    "id", FOUNDRY_JSON_NODE_GET_STRING (&id),
                                    "format", FOUNDRY_JSON_NODE_GET_STRING (&format),
                                  "}"))
    return g_error_new_literal (G_IO_ERROR,
                                G_IO_ERROR_FAILED,
                                "Failed");

  /* TODO: expand format string */

  return g_error_new (G_IO_ERROR,
                      G_IO_ERROR_FAILED,
                      "%s: %s", id, format);
}

/**
 * foundry_dap_protocol_unwrap_error:
 *
 * Returns: (transfer full):
 */
DexFuture *
foundry_dap_protocol_unwrap_error (DexFuture *completed,
                                   gpointer   user_data)
{
  g_autoptr(JsonNode) reply = NULL;
  g_autoptr(GError) error = NULL;

  if ((reply = dex_await_boxed (dex_ref (completed), &error)))
    {
      if (foundry_dap_protocol_has_error (reply))
        return dex_future_new_for_error (foundry_dap_protocol_extract_error (reply));
    }

  return dex_ref (completed);
}
