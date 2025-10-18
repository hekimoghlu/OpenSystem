/*
 * ggit-message.h
 * This file is part of libgit2-glib
 *
 * Copyright (C) 2013 - Jesse van den Kieboom
 *
 * libgit2-glib is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * libgit2-glib is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with libgit2-glib. If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef __GGIT_MESSAGE_H__
#define __GGIT_MESSAGE_H__

#include <glib.h>

G_BEGIN_DECLS

gchar *ggit_message_prettify (const gchar *message,
                              gboolean     strip_comments,
                              gchar        comment_char);

G_END_DECLS

#endif /* __GGIT_MESSAGE_H__ */

/* ex:set ts=8 noet: */
