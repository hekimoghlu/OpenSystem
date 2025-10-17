/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, February 14, 2024.
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
#pragma once

#if OS(DARWIN)
#include <wtf/MachSendRight.h>
#endif

#if USE(UNIX_DOMAIN_SOCKETS)
#include <wtf/unix/UnixFileDescriptor.h>
#endif

namespace IPC {

// IPC::Attachment is a type representing objects that cannot be transferred as data,
// rather they are transferred via operating system cross-process communication primitives.
#if OS(DARWIN)
using Attachment = MachSendRight;
#elif OS(WINDOWS)
struct Attachment { }; // Windows does not need attachments at the moment.
#elif USE(UNIX_DOMAIN_SOCKETS)
using Attachment = UnixFileDescriptor;
#else
#error Unsupported platform
#endif

} // namespace IPC
