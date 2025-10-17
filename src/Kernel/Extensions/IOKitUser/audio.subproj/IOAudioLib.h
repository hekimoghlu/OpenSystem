/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, September 11, 2024.
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
/*
 * Copyright (c) 1998-2000 Apple Computer, Inc. All rights reserved.
 */

/*!
 * @header IOAudioLib
 * C interface to IOAudio functions
 */

#ifndef _IOAUDIOLIB_H
#define _IOAUDIOLIB_H

#if TARGET_OS_OSX
#include <IOKit/audio/IOAudioTypes.h>
#endif

#if 0

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * @function IOAudioIsOutput
 * @abstract Determines if the audio stream is an output stream
 * @param service
 * @param out
 * @result kern_return_t
 */
kern_return_t IOAudioIsOutput(io_service_t service, int *out);

/*!
 * @function IOAudioFlush
 * @abstract Indicate the position at which the audio stream can be stopped.
 * @param connect the audio stream
 * @param end the position
 * @result kern_return_t
 */
kern_return_t IOAudioFlush(io_connect_t connect, IOAudioStreamPosition *end);

/*!
 * @function IOAudioSetErase
 * @abstract Set autoerase flag, returns old value
 * @param connect the audio stream
 * @param erase true to turn off, false otherwise
 * @param oldVal previous value
 * @result kern_return_t
 */
kern_return_t IOAudioSetErase(io_connect_t connect, int erase, int *oldVal);

#ifdef __cplusplus
}
#endif

#endif /* 0 */

#endif /* ! _IOAUDIOLIB_H */
