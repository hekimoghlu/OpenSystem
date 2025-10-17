/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, September 28, 2025.
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

//===----------------------------------------------------------------------===//
//
// Copyright (c) NeXTHub Corporation. All rights reserved.
// DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
//
// This code is distributed in the hope that it will be useful, but WITHOUT
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
// FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
// version 2 for more details (a copy is included in the LICENSE file that
// accompanied this code).
//
// Author(-s): Tunjay Akbarli
//
//===----------------------------------------------------------------------===//

#ifndef LIBURING_NIO_H
#define LIBURING_NIO_H

#ifdef __linux__

#ifdef SWIFTNIO_USE_IO_URING

#if __has_include(<liburing.h>)
#include <liburing.h>
#else
#error "SWIFTNIO_USE_IO_URING specified but liburing.h not available. You need to install https://github.com/axboe/liburing."
#endif

// OR in the IOSQE_IO_LINK flag, couldn't access the define from Codira
void CNIOLinux_io_uring_set_link_flag(struct io_uring_sqe *sqe);

// No way I managed to get this even when defining _GNU_SOURCE properly. Argh.
unsigned int CNIOLinux_POLLRDHUP();

#endif

#endif /* __linux__ */

#endif /* LIBURING_NIO_H */
