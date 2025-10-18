/*
 * ggit-diff-binary-file.h
 * This file is part of libgit2-glib
 *
 * Copyright (C) 2015 - Ignacio Casal Quinteiro
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

#ifndef __GGIT_DIFF_BINARY_FILE_H__
#define __GGIT_DIFF_BINARY_FILE_H__

#include <git2.h>

#include "ggit-types.h"

G_BEGIN_DECLS

#define GGIT_TYPE_DIFF_BINARY_FILE       (ggit_diff_binary_file_get_type ())
#define GGIT_DIFF_BINARY_FILE(obj)       ((GgitDiffBinaryFile *)obj)

GType                    ggit_diff_binary_file_get_type          (void) G_GNUC_CONST;

GgitDiffBinaryFile      *_ggit_diff_binary_file_wrap             (const git_diff_binary_file *file);

GgitDiffBinaryFile      *ggit_diff_binary_file_ref               (GgitDiffBinaryFile         *file);
void                     ggit_diff_binary_file_unref             (GgitDiffBinaryFile         *file);

GgitDiffBinaryType       ggit_diff_binary_file_get_binary_type   (GgitDiffBinaryFile         *file);

const guint8            *ggit_diff_binary_file_get_data          (GgitDiffBinaryFile         *file,
                                                                  gsize                      *size);

gsize                    ggit_diff_binary_file_get_inflated_size (GgitDiffBinaryFile         *file);

G_END_DECLS

#endif /* __GGIT_DIFF_BINARY_FILE_H__ */

/* ex:set ts=8 noet: */
