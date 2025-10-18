/*
 * Copyright (C) 2014 Carlos Garnacho  <carlosg@gnome.org>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin Street, Fifth Floor,
 * Boston, MA  02110-1301, USA.
 */

#ifndef __LIBTRACKER_MINER_DECORATOR_H__
#define __LIBTRACKER_MINER_DECORATOR_H__

#include <tracker-common.h>

#include "tracker-extract.h"
#include "tracker-extract-persistence.h"
#include "utils/tracker-extract.h"

G_BEGIN_DECLS

#define TRACKER_TYPE_DECORATOR (tracker_decorator_get_type())
G_DECLARE_FINAL_TYPE (TrackerDecorator,
                      tracker_decorator,
                      TRACKER, DECORATOR,
                      TrackerMiner)

TrackerDecorator * tracker_decorator_new (TrackerSparqlConnection   *connection,
                                          TrackerExtract            *extract,
                                          TrackerExtractPersistence *persistence);

void tracker_decorator_set_priority_graphs (TrackerDecorator    *decorator,
                                            const gchar * const *graphs);

void tracker_decorator_set_throttled (TrackerDecorator *decorator,
                                      gboolean          throttled);

void tracker_decorator_check_unextracted (TrackerDecorator *decorator);

G_END_DECLS

#endif /* __LIBTRACKER_MINER_DECORATOR_H__ */
