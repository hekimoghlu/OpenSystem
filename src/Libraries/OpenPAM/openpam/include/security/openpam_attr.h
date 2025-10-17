/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, December 24, 2021.
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
#ifndef SECURITY_PAM_ATTRIBUTES_H_INCLUDED
#define SECURITY_PAM_ATTRIBUTES_H_INCLUDED

/* GCC attributes */
#if defined(__GNUC__) && defined(__GNUC_MINOR__) && !defined(__STRICT_ANSI__)
# define OPENPAM_GNUC_PREREQ(maj, min) \
        ((__GNUC__ << 16) + __GNUC_MINOR__ >= ((maj) << 16) + (min))
#else
# define OPENPAM_GNUC_PREREQ(maj, min) 0
#endif

#if OPENPAM_GNUC_PREREQ(2,5)
# define OPENPAM_FORMAT(params) __attribute__((__format__ params))
#else
# define OPENPAM_FORMAT(params)
#endif

#if OPENPAM_GNUC_PREREQ(3,3)
# define OPENPAM_NONNULL(params) __attribute__((__nonnull__ params))
#else
# define OPENPAM_NONNULL(params)
#endif

#endif /* !SECURITY_PAM_ATTRIBUTES_H_INCLUDED */
