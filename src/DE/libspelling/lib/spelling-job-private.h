/* spelling-job-private.h
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

#pragma once

#include <gio/gio.h>

#include "spelling-dictionary-internal.h"

G_BEGIN_DECLS

typedef struct _SpellingMistake
{
  guint offset;
  guint length;
} SpellingMistake;

#define SPELLING_TYPE_JOB (spelling_job_get_type())

G_DECLARE_FINAL_TYPE (SpellingJob, spelling_job, SPELLING, JOB, GObject)

SpellingJob     *spelling_job_new           (SpellingDictionary   *dictionary,
                                             PangoLanguage        *language);
void             spelling_job_discard       (SpellingJob          *self);
void             spelling_job_run           (SpellingJob          *self,
                                             GAsyncReadyCallback   callback,
                                             gpointer              user_data);
void             spelling_job_run_finish    (SpellingJob          *self,
                                             GAsyncResult         *result,
                                             SpellingBoundary    **fragments,
                                             guint                *n_fragments,
                                             SpellingMistake     **mistakes,
                                             guint                *n_mistakes);
void             spelling_job_run_sync      (SpellingJob          *self,
                                             SpellingBoundary    **fragments,
                                             guint                *n_fragments,
                                             SpellingMistake     **mistakes,
                                             guint                *n_mistakes);
void             spelling_job_add_fragment  (SpellingJob          *self,
                                             GBytes               *bytes,
                                             guint                 position,
                                             guint                 length);
void             spelling_job_notify_delete (SpellingJob          *self,
                                             guint                 position,
                                             guint                 length);
void             spelling_job_notify_insert (SpellingJob          *self,
                                             guint                 position,
                                             guint                 length);
void             spelling_job_invalidate    (SpellingJob          *self,
                                             guint                 position,
                                             guint                 length);

G_END_DECLS
