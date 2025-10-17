/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, December 18, 2022.
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
#ifndef __SOUND_SFNT_INFO_H
#define __SOUND_SFNT_INFO_H
#include <sound/asound.h>
#ifdef SNDRV_BIG_ENDIAN
#define SNDRV_OSS_PATCHKEY(id) (0xfd00 | id)
#else
#define SNDRV_OSS_PATCHKEY(id) ((id << 8) | 0xfd)
#endif
struct soundfont_patch_info {
  unsigned short key;
#define SNDRV_OSS_SOUNDFONT_PATCH SNDRV_OSS_PATCHKEY(0x07)
  short device_no;
  unsigned short sf_id;
  short optarg;
  int len;
  short type;
#define SNDRV_SFNT_LOAD_INFO 0
#define SNDRV_SFNT_LOAD_DATA 1
#define SNDRV_SFNT_OPEN_PATCH 2
#define SNDRV_SFNT_CLOSE_PATCH 3
#define SNDRV_SFNT_REPLACE_DATA 5
#define SNDRV_SFNT_MAP_PRESET 6
#define SNDRV_SFNT_PROBE_DATA 8
#define SNDRV_SFNT_REMOVE_INFO 9
  short reserved;
};
#define SNDRV_SFNT_PATCH_NAME_LEN 32
struct soundfont_open_parm {
  unsigned short type;
#define SNDRV_SFNT_PAT_TYPE_MISC 0
#define SNDRV_SFNT_PAT_TYPE_GUS 6
#define SNDRV_SFNT_PAT_TYPE_MAP 7
#define SNDRV_SFNT_PAT_LOCKED 0x100
#define SNDRV_SFNT_PAT_SHARED 0x200
  short reserved;
  char name[SNDRV_SFNT_PATCH_NAME_LEN];
};
struct soundfont_voice_parm {
  unsigned short moddelay;
  unsigned short modatkhld;
  unsigned short moddcysus;
  unsigned short modrelease;
  short modkeyhold, modkeydecay;
  unsigned short voldelay;
  unsigned short volatkhld;
  unsigned short voldcysus;
  unsigned short volrelease;
  short volkeyhold, volkeydecay;
  unsigned short lfo1delay;
  unsigned short lfo2delay;
  unsigned short pefe;
  unsigned short fmmod;
  unsigned short tremfrq;
  unsigned short fm2frq2;
  unsigned char cutoff;
  unsigned char filterQ;
  unsigned char chorus;
  unsigned char reverb;
  unsigned short reserved[4];
};
struct soundfont_voice_info {
  unsigned short sf_id;
  unsigned short sample;
  int start, end;
  int loopstart, loopend;
  short rate_offset;
  unsigned short mode;
#define SNDRV_SFNT_MODE_ROMSOUND 0x8000
#define SNDRV_SFNT_MODE_STEREO 1
#define SNDRV_SFNT_MODE_LOOPING 2
#define SNDRV_SFNT_MODE_NORELEASE 4
#define SNDRV_SFNT_MODE_INIT_PARM 8
  short root;
  short tune;
  unsigned char low, high;
  unsigned char vellow, velhigh;
  signed char fixkey, fixvel;
  signed char pan, fixpan;
  short exclusiveClass;
  unsigned char amplitude;
  unsigned char attenuation;
  short scaleTuning;
  struct soundfont_voice_parm parm;
  unsigned short sample_mode;
};
struct soundfont_voice_rec_hdr {
  unsigned char bank;
  unsigned char instr;
  char nvoices;
  char write_mode;
#define SNDRV_SFNT_WR_APPEND 0
#define SNDRV_SFNT_WR_EXCLUSIVE 1
#define SNDRV_SFNT_WR_REPLACE 2
};
struct soundfont_sample_info {
  unsigned short sf_id;
  unsigned short sample;
  int start, end;
  int loopstart, loopend;
  int size;
  short dummy;
  unsigned short mode_flags;
#define SNDRV_SFNT_SAMPLE_8BITS 1
#define SNDRV_SFNT_SAMPLE_UNSIGNED 2
#define SNDRV_SFNT_SAMPLE_NO_BLANK 4
#define SNDRV_SFNT_SAMPLE_SINGLESHOT 8
#define SNDRV_SFNT_SAMPLE_BIDIR_LOOP 16
#define SNDRV_SFNT_SAMPLE_STEREO_LEFT 32
#define SNDRV_SFNT_SAMPLE_STEREO_RIGHT 64
#define SNDRV_SFNT_SAMPLE_REVERSE_LOOP 128
  unsigned int truesize;
};
struct soundfont_voice_map {
  int map_bank, map_instr, map_key;
  int src_bank, src_instr, src_key;
};
#define SNDRV_EMUX_HWDEP_NAME "Emux WaveTable"
#define SNDRV_EMUX_VERSION ((1 << 16) | (0 << 8) | 0)
struct snd_emux_misc_mode {
  int port;
  int mode;
  int value;
  int value2;
};
#define SNDRV_EMUX_IOCTL_VERSION _IOR('H', 0x80, unsigned int)
#define SNDRV_EMUX_IOCTL_LOAD_PATCH _IOWR('H', 0x81, struct soundfont_patch_info)
#define SNDRV_EMUX_IOCTL_RESET_SAMPLES _IO('H', 0x82)
#define SNDRV_EMUX_IOCTL_REMOVE_LAST_SAMPLES _IO('H', 0x83)
#define SNDRV_EMUX_IOCTL_MEM_AVAIL _IOW('H', 0x84, int)
#define SNDRV_EMUX_IOCTL_MISC_MODE _IOWR('H', 0x84, struct snd_emux_misc_mode)
#endif
