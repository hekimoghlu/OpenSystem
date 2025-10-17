/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, April 15, 2025.
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
#ifndef tcpdump_interface_h
#define tcpdump_interface_h

#ifdef HAVE_OS_PROTO_H
#include "os-proto.h"
#endif

#include "funcattrs.h"

#include <stdarg.h>

#if HAVE_STDINT_H
#include <stdint.h>
#endif

#ifndef HAVE_STRLCAT
extern size_t strlcat(char *, const char *, size_t);
#endif
#ifndef HAVE_STRLCPY
extern size_t strlcpy(char *, const char *, size_t);
#endif

#ifndef HAVE_STRDUP
extern char *strdup(const char *);
#endif

#ifndef HAVE_STRSEP
extern char *strsep(char **, const char *);
#endif

#endif

extern char *program_name;	/* used to generate self-identifying messages */

#include <pcap.h>

#ifndef HAVE_BPF_DUMP
struct bpf_program;

extern void bpf_dump(const struct bpf_program *, int);

#endif
