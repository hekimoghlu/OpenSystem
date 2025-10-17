/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 12, 2022.
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
#ifndef _UAPI_LINUX_SOUND_H
#define _UAPI_LINUX_SOUND_H
#include <linux/fs.h>
#define SND_DEV_CTL 0
#define SND_DEV_SEQ 1
#define SND_DEV_MIDIN 2
#define SND_DEV_DSP 3
#define SND_DEV_AUDIO 4
#define SND_DEV_DSP16 5
#define SND_DEV_UNUSED 6
#define SND_DEV_AWFM 7
#define SND_DEV_SEQ2 8
#define SND_DEV_SYNTH 9
#define SND_DEV_DMFM 10
#define SND_DEV_UNKNOWN11 11
#define SND_DEV_ADSP 12
#define SND_DEV_AMIDI 13
#define SND_DEV_ADMMIDI 14
#endif
