/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, October 23, 2023.
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
#ifndef _SYS_PREOSLOG_H_
#define _SYS_PREOSLOG_H_

#include <sys/cdefs.h>

__BEGIN_DECLS

#define PREOSLOG_MAGIC 'LSOP'
#define PREOSLOG_SYSCTL "kern.preoslog"

typedef uint8_t preoslog_source_t;
enum {
	PREOSLOG_SOURCE_IBOOT = 0,
	PREOSLOG_SOURCE_MACEFI,
	PREOSLOG_SOURCE_MAX
};

/*
 * Any change to this structure must be reflected in boot loader (iboot / xnu SDK header) and vice versa.
 * Beware: if you remove __attribute__((packed)) here, then sizeof() on this structure will return 16.
 * However, with or without __attribute__((packed)), offset_of(preoslog_header_t, data) will always return 14.
 */

typedef struct  __attribute__((packed)) {
	uint32_t magic; /* PREOGLOS_MAGIC if valid */
	uint32_t size; /* Size of the preoslog buffer including the header */
	uint32_t offset; /* Write pointer. Indicates where in the buffer new log entry would go */
	preoslog_source_t source; /* Indicates who filled in the buffer (e.g. iboot vs MacEFI) */
	uint8_t wrapped; /* If equal to 1, the preoslog ring buffer wrapped at least once */
	char data[]; /* log buffer */
} preoslog_header_t;

__END_DECLS

#endif  /* !_SYS_PREOSLOG_H_ */
