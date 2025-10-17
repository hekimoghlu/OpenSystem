/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 29, 2022.
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
#ifndef AUDIO_DEVICE_AUDIO_DEVICE_CONFIG_H_
#define AUDIO_DEVICE_AUDIO_DEVICE_CONFIG_H_

// Enumerators
//
enum { GET_MIC_VOLUME_INTERVAL_MS = 1000 };

// Platform specifics
//
#if defined(_WIN32)
#if (_MSC_VER >= 1400)
#if !defined(WEBRTC_DUMMY_FILE_DEVICES)
// Windows Core Audio is the default audio layer in Windows.
// Only supported for VS 2005 and higher.
#define WEBRTC_WINDOWS_CORE_AUDIO_BUILD
#endif
#endif
#endif

#endif  // AUDIO_DEVICE_AUDIO_DEVICE_CONFIG_H_
