/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, September 26, 2023.
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
#ifndef _COLLATE_H_
#define	_COLLATE_H_

#include <sys/cdefs.h>
#ifndef __LIBC__
#include <sys/types.h>
#endif /* !__LIBC__ */
#include <limits.h>

#define STR_LEN 10
#define TABLE_SIZE 100
#define COLLATE_VERSION    "1.0\n"
#define COLLATE_VERSION1_1 "1.1\n"
#define COLLATE_VERSION1_1A "1.1A\n"
/* see discussion in string/FreeBSD/strxfrm for this value */
#define COLLATE_MAX_PRIORITY ((1 << 24) - 1)

#define DIRECTIVE_UNDEF 0x00
#define DIRECTIVE_FORWARD 0x01
#define DIRECTIVE_BACKWARD 0x02
#define DIRECTIVE_POSITION 0x04

#define DIRECTIVE_DIRECTION_MASK (DIRECTIVE_FORWARD | DIRECTIVE_BACKWARD)

#define COLLATE_SUBST_DUP 0x0001

#define IGNORE_EQUIV_CLASS 1

struct __collate_st_info {
	__uint8_t directive[COLL_WEIGHTS_MAX];
	__uint8_t flags;
#if __DARWIN_BYTE_ORDER == __DARWIN_LITTLE_ENDIAN
	__uint8_t directive_count:4;
	__uint8_t chain_max_len:4;
#else
	__uint8_t chain_max_len:4;
	__uint8_t directive_count:4;
#endif
	__int32_t undef_pri[COLL_WEIGHTS_MAX];
	__int32_t subst_count[COLL_WEIGHTS_MAX];
	__int32_t chain_count;
	__int32_t large_pri_count;
};

struct __collate_st_char_pri {
	__int32_t pri[COLL_WEIGHTS_MAX];
};
struct __collate_st_chain_pri {
	__darwin_wchar_t str[STR_LEN];
	__int32_t pri[COLL_WEIGHTS_MAX];
};
struct __collate_st_large_char_pri {
	__int32_t val;
	struct __collate_st_char_pri pri;
};
struct __collate_st_subst {
	__int32_t val;
	__darwin_wchar_t str[STR_LEN];
};

#ifndef __LIBC__
extern int __collate_load_error;
extern int __collate_substitute_nontrivial;
#define __collate_char_pri_table (*__collate_char_pri_table_ptr)
extern struct __collate_st_char_pri __collate_char_pri_table[UCHAR_MAX + 1];
extern struct __collate_st_chain_pri *__collate_chain_pri_table;
extern __int32_t *__collate_chain_equiv_table;
extern struct __collate_st_info __collate_info;
#endif /* !__LIBC__ */

__BEGIN_DECLS
#ifdef __LIBC__
__darwin_wchar_t	*__collate_mbstowcs(const char *, locale_t);
__darwin_wchar_t	*__collate_wcsdup(const __darwin_wchar_t *);
__darwin_wchar_t	*__collate_substitute(const __darwin_wchar_t *, int, locale_t);
int	__collate_load_tables(const char *, locale_t);
void	__collate_lookup_l(const __darwin_wchar_t *, int *, int *, int *, locale_t);
void	__collate_lookup_which(const __darwin_wchar_t *, int *, int *, int, locale_t);
void	__collate_xfrm(const __darwin_wchar_t *, __darwin_wchar_t **, locale_t);
int	__collate_range_cmp(__darwin_wchar_t, __darwin_wchar_t, locale_t);
size_t	__collate_collating_symbol(__darwin_wchar_t *, size_t, const char *, size_t, __darwin_mbstate_t *, locale_t);
int	__collate_equiv_class(const char *, size_t, __darwin_mbstate_t *, locale_t);
size_t	__collate_equiv_match(int, __darwin_wchar_t *, size_t, __darwin_wchar_t, const char *, size_t, __darwin_mbstate_t *, size_t *, locale_t);
#else /* !__LIBC__ */
void	__collate_lookup(const unsigned char *, int *, int *, int *);
#endif /* __LIBC__ */
#ifdef COLLATE_DEBUG
void	__collate_print_tables(void);
#endif
__END_DECLS

#endif /* !_COLLATE_H_ */
