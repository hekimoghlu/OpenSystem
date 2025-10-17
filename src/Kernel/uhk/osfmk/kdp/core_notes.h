/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 7, 2021.
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
/* Common LC_NOTE defintions for core files. */
#ifndef _CORE_NOTES_H_
#define _CORE_NOTES_H_

/*
 * Format of the "main bin spec" LC_NOTE payload as expected by LLDB
 */
#define MAIN_BIN_SPEC_DATA_OWNER "main bin spec"

typedef struct main_bin_spec_note {
	uint32_t version;       // currently 1
	uint32_t type;          // 0 == unspecified, 1 == kernel, 2 == user process, 3 == standalone (ie FW)
	uint64_t address;       // UINT64_MAX if address not specified
	uuid_t   uuid;          // all zeros if uuid not specified
	uint32_t log2_pagesize; // process page size in log base 2, e.g. 4k pages are 12. 0 for unspecified
	uint32_t unused;        // leave set to 0
} __attribute__((packed)) main_bin_spec_note_t;

#define MAIN_BIN_SPEC_VERSION 1
#define MAIN_BIN_SPEC_TYPE_KERNEL 1
#define MAIN_BIN_SPEC_TYPE_USER 2
#define MAIN_BIN_SPEC_TYPE_STANDALONE 3


/*
 * Format of the "load binary" LC_NOTE payload as expected by LLDB
 */
#define LOAD_BINARY_SPEC_DATA_OWNER "load binary"

#define LOAD_BINARY_NAME_BUF_SIZE 32
typedef struct load_binary_spec_note {
	uint32_t version;    // currently 1
	uuid_t   uuid;       // all zeroes if uuid not specified
	uint64_t address;    // virtual address where the macho is loaded, UINT64_MAX if unavail
	uint64_t slide;      // UINT64_MAX if slide not specified/unknown
	                     // 0 if there is no slide (the binary loaded at
	                     // the vmaddr in the file)
	/*
	 * name_cstring must be a NUL terminated C string, or empty ('\0')
	 * if unavailable.  NOTE: lldb's spec does not specify a length
	 * for the name, it just wants a NUL terminated string. But we
	 * specify a (maximum) length to avoid notes with dynamic length.
	 */
	char     name_cstring[LOAD_BINARY_NAME_BUF_SIZE];
} __attribute__((packed)) load_binary_spec_note_t;

#define LOAD_BINARY_SPEC_VERSION 1

/*
 * Format of the "addrable bits" LC_NOTE payload as expected by LLDB.
 */
#define ADDRABLE_BITS_DATA_OWNER "addrable bits"

typedef struct addrable_bits_note {
	uint32_t version;            // CURRENTLY 3
	uint32_t addressing_bits;    // # of bits in use for addressing
	uint64_t unused;             // zeroed
} __attribute__((packed)) addrable_bits_note_t;

#define ADDRABLE_BITS_VER 3


#define PANIC_CONTEXT_DATA_OWNER "panic context"

typedef struct panic_context_note {
	uuid_string_t kernel_uuid_string;
} __attribute__((packed)) panic_context_note_t;

#endif /* _CORE_NOTES_H_ */
