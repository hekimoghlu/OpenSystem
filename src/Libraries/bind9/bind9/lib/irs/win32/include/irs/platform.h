/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 17, 2024.
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
/*! \file */

#ifndef IRS_PLATFORM_H
#define IRS_PLATFORM_H 1

/*****
 ***** Platform-dependent defines.
 *****/

#ifndef IRS_PLATFORM_USEDECLSPEC
#define LIBIRS_EXTERNAL_DATA
#else
#ifdef LIBIRS_EXPORTS
#define LIBIRS_EXTERNAL_DATA __declspec(dllexport)
#else
#define LIBIRS_EXTERNAL_DATA __declspec(dllimport)
#endif
#endif

/*
 * Tell Emacs to use C mode on this file.
 * Local Variables:
 * mode: c
 * End:
 */

#endif /* IRS_PLATFORM_H */
