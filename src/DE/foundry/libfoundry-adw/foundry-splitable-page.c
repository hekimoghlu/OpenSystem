/* foundry-splitable-page.c
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

#include "foundry-splitable-page.h"

G_DEFINE_INTERFACE (FoundrySplitablePage, foundry_splitable_page, FOUNDRY_TYPE_PAGE)

static void
foundry_splitable_page_default_init (FoundrySplitablePageInterface *iface)
{
}

/**
 * foundry_splitable_page_split:
 * @self: a [iface@FoundryAdw.SplitablePage]
 *
 * Returns: (transfer full): the new split page
 */
FoundryPage *
foundry_splitable_page_split (FoundrySplitablePage *self)
{
  g_return_val_if_fail (FOUNDRY_IS_SPLITABLE_PAGE (self), NULL);

  return FOUNDRY_SPLITABLE_PAGE_GET_IFACE (self)->split (self);
}
