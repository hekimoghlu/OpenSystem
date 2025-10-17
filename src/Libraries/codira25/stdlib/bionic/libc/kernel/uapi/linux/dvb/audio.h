/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, May 17, 2022.
 *
 * Licensed under the Apache License, Version 2.0 (the ""License"");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at:
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an ""AS IS"" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Please contact NeXTHub Corporation, 651 N Broad St, Suite 201, 
 * Middletown, DE 19709, New Castle County, USA.
 *
 */
#ifndef _DVBAUDIO_H_
#define _DVBAUDIO_H_
#include <linux/types.h>
typedef enum {
  AUDIO_SOURCE_DEMUX,
  AUDIO_SOURCE_MEMORY
} audio_stream_source_t;
typedef enum {
  AUDIO_STOPPED,
  AUDIO_PLAYING,
  AUDIO_PAUSED
} audio_play_state_t;
typedef enum {
  AUDIO_STEREO,
  AUDIO_MONO_LEFT,
  AUDIO_MONO_RIGHT,
  AUDIO_MONO,
  AUDIO_STEREO_SWAPPED
} audio_channel_select_t;
typedef struct audio_mixer {
  unsigned int volume_left;
  unsigned int volume_right;
} audio_mixer_t;
typedef struct audio_status {
  int AV_sync_state;
  int mute_state;
  audio_play_state_t play_state;
  audio_stream_source_t stream_source;
  audio_channel_select_t channel_select;
  int bypass_mode;
  audio_mixer_t mixer_state;
} audio_status_t;
#define AUDIO_CAP_DTS 1
#define AUDIO_CAP_LPCM 2
#define AUDIO_CAP_MP1 4
#define AUDIO_CAP_MP2 8
#define AUDIO_CAP_MP3 16
#define AUDIO_CAP_AAC 32
#define AUDIO_CAP_OGG 64
#define AUDIO_CAP_SDDS 128
#define AUDIO_CAP_AC3 256
#define AUDIO_STOP _IO('o', 1)
#define AUDIO_PLAY _IO('o', 2)
#define AUDIO_PAUSE _IO('o', 3)
#define AUDIO_CONTINUE _IO('o', 4)
#define AUDIO_SELECT_SOURCE _IO('o', 5)
#define AUDIO_SET_MUTE _IO('o', 6)
#define AUDIO_SET_AV_SYNC _IO('o', 7)
#define AUDIO_SET_BYPASS_MODE _IO('o', 8)
#define AUDIO_CHANNEL_SELECT _IO('o', 9)
#define AUDIO_GET_STATUS _IOR('o', 10, audio_status_t)
#define AUDIO_GET_CAPABILITIES _IOR('o', 11, unsigned int)
#define AUDIO_CLEAR_BUFFER _IO('o', 12)
#define AUDIO_SET_ID _IO('o', 13)
#define AUDIO_SET_MIXER _IOW('o', 14, audio_mixer_t)
#define AUDIO_SET_STREAMTYPE _IO('o', 15)
#define AUDIO_BILINGUAL_CHANNEL_SELECT _IO('o', 20)
#endif
