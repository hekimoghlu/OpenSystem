/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, November 22, 2022.
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

#if defined( GXVALID_H_ )
#pragma warn -aus /* too many unevaluated variables in gxvalid */
#endif

#ifndef ATARI_H
#define ATARI_H

#pragma warn -stu

/* PureC doesn't like 32bit enumerations */

#ifndef FT_IMAGE_TAG
#define FT_IMAGE_TAG( value, _x1, _x2, _x3, _x4 ) value
#endif /* FT_IMAGE_TAG */

#ifndef FT_ENC_TAG
#define FT_ENC_TAG( value, a, b, c, d ) value
#endif /* FT_ENC_TAG */

#endif /* ATARI_H */
