/*
 * mks-speaker.h
 *
 * Copyright 2023 Christian Hergert <chergert@redhat.com>
 *
 * This library is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as
 * published by the Free Software Foundation; either version 2.1 of the
 * License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 */

#pragma once

#include "mks-device.h"

G_BEGIN_DECLS

#define MKS_TYPE_SPEAKER            (mks_speaker_get_type ())
#define MKS_SPEAKER(obj)            (G_TYPE_CHECK_INSTANCE_CAST ((obj), MKS_TYPE_SPEAKER, MksSpeaker))
#define MKS_SPEAKER_CONST(obj)      (G_TYPE_CHECK_INSTANCE_CAST ((obj), MKS_TYPE_SPEAKER, MksSpeaker const))
#define MKS_SPEAKER_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST ((klass),  MKS_TYPE_SPEAKER, MksSpeakerClass))
#define MKS_IS_SPEAKER(obj)         (G_TYPE_CHECK_INSTANCE_TYPE ((obj), MKS_TYPE_SPEAKER))
#define MKS_IS_SPEAKER_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE ((klass),  MKS_TYPE_SPEAKER))
#define MKS_SPEAKER_GET_CLASS(obj)  (G_TYPE_INSTANCE_GET_CLASS ((obj),  MKS_TYPE_SPEAKER, MksSpeakerClass))

typedef struct _MksSpeakerClass MksSpeakerClass;

MKS_AVAILABLE_IN_ALL
GType    mks_speaker_get_type  (void) G_GNUC_CONST;
MKS_AVAILABLE_IN_ALL
gboolean mks_speaker_get_muted (MksSpeaker *self);
MKS_AVAILABLE_IN_ALL
void     mks_speaker_set_muted (MksSpeaker *self,
                                gboolean    muted);

G_DEFINE_AUTOPTR_CLEANUP_FUNC (MksSpeaker, g_object_unref)

G_END_DECLS
