/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, November 3, 2021.
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
#ifndef _SYS_LINKER_SET_H_
#define _SYS_LINKER_SET_H_

#include <sys/appleapiopts.h>
#if !defined(KERNEL) || defined(__APPLE_API_PRIVATE)

/*
 * The following macros are used to declare global sets of objects, which
 * are collected by the linker into a `linker set' as defined below.
 * For Mach-O, this is done by constructing a separate segment inside the
 * __DATA_CONST section for each set.  The contents of this segment are an array
 * of pointers to the objects in the set.
 *
 * Note that due to limitations of the Mach-O format, there cannot
 * be more than 255 sections in a segment, so linker set usage should be
 * conserved.  Set names may not exceed 16 characters.
 */

#ifdef KERNEL
# include <mach-o/loader.h>
# include <libkern/kernel_mach_header.h>

# define MACH_HEADER_TYPE kernel_mach_header_t
# define GETSECTIONDATA_VARIANT getsectdatafromheader
# define SECTDATA_SIZE_TYPE unsigned long
# define MH_EXECUTE_HEADER &_mh_execute_header
# define IMAGE_SLIDE_CORRECT 0
#else
# include <mach-o/ldsyms.h>
# include <mach-o/getsect.h>
# include <mach-o/loader.h>
# include <mach-o/dyld.h>
# include <mach-o/dyld_priv.h>
# include <crt_externs.h>

# if __LP64__
#  define MACH_HEADER_TYPE struct mach_header_64
#  define GETSECTIONDATA_VARIANT getsectdatafromheader_64
#  define SECTDATA_SIZE_TYPE uint64_t
#  define MH_EXECUTE_HEADER _NSGetMachExecuteHeader()
# else
#  define MACH_HEADER_TYPE struct mach_header
#  define GETSECTIONDATA_VARIANT getsectdatafromheader
#  define SECTDATA_SIZE_TYPE uint32_t
#  define MH_EXECUTE_HEADER _NSGetMachExecuteHeader()
# endif
#endif

#if __LP64__ && !(__x86_64__ || __i386__)
# define LINKER_SET_ENTRY_PACKED
# define LINKER_SET_SEGMENT __DATA_CONST
# define LINKER_SET_SEGMENT_CSTR "__DATA_CONST"
#else
# define LINKER_SET_ENTRY_PACKED __attribute__((packed))
# define LINKER_SET_SEGMENT __DATA
# define LINKER_SET_SEGMENT_CSTR "__DATA"
#endif

/*
 * Private macros, not to be used outside this header file.
 *
 * The objective of this macro stack is to produce the following output,
 * given SET and SYM as arguments:
 *
 *  void const * __set_SET_sym_SYM __attribute__((section("__DATA_CONST,SET"))) = & SYM
 */

/* Wrap entries in a type that can be blacklisted from KASAN */
struct linker_set_entry {
	void *ptr;
} LINKER_SET_ENTRY_PACKED;

#ifdef __LS_VA_STRINGIFY__
#  undef __LS_VA_STRINGIFY__
#endif
#ifdef __LS_VA_STRCONCAT__
#  undef __LS_VA_STRCONCAT__
#endif
#define __LS_VA_STRINGIFY(_x ...)        #_x
#define __LS_VA_STRCONCAT(_x, _y)        __LS_VA_STRINGIFY(_x,_y)
#define __LINKER_MAKE_SET(_set, _sym)                                   \
	/*__unused*/ /*static*/ const struct linker_set_entry /*const*/ __set_##_set##_sym_##_sym               \
	__attribute__ ((section(__LS_VA_STRCONCAT(LINKER_SET_SEGMENT,_set)),used)) = { (void *)&_sym }
/* the line above is very fragile - if your compiler breaks linker sets,
 *  just play around with "static", "const", "used" etc. :-) */

/*
 * Public macros.
 */
#define LINKER_SET_ENTRY(_set, _sym)    __LINKER_MAKE_SET(_set, _sym)

/*
 * FreeBSD compatibility.
 */
#ifdef __APPLE_API_OBSOLETE
# define TEXT_SET(_set, _sym)   __LINKER_MAKE_SET(_set, _sym)
# define DATA_SET(_set, _sym)   __LINKER_MAKE_SET(_set, _sym)
# define BSS_SET(_set, _sym)    __LINKER_MAKE_SET(_set, _sym)
# define ABS_SET(_set, _sym)    __LINKER_MAKE_SET(_set, _sym)
# define SET_ENTRY(_set, _sym)  __LINKER_MAKE_SET(_set, _sym)
#endif /* __APPLE_API_OBSOLETE */

/*
 * Extended linker set API.
 *
 * Since linker sets are per-object-file, and we may have multiple
 * object files, we need to be able to specify which object's set
 * to scan.
 *
 * The set itself is a contiguous array of pointers to the objects
 * within the set.
 */

/*
 * Public interface.
 *
 * void **LINKER_SET_OBJECT_BEGIN(_object, _set)
 *	Preferred interface to linker_set_object_begin(), takes set name unquoted.
 * void **LINKER_SET_OBJECT_LIMIT(_object, _set)
 *	Preferred interface to linker_set_object_begin(), takes set name unquoted.
 * LINKER_SET_OBJECT_FOREACH(_object, (set_member_type **)_pvar, _cast, _set)
 *	Iterates over the members of _set within _object.  Since the set contains
 *	pointers to its elements, for a set of elements of type etyp, _pvar must
 *	be (etyp **).
 * LINKER_SET_FOREACH((set_member_type **)_pvar, _cast, _set)
 *
 * Example of _cast: For the _pvar "struct sysctl_oid **oidpp", _cast would be
 *                   "struct sysctl_oid **"
 *
 */

#define LINKER_SET_OBJECT_BEGIN(_object, _set)  __linker_set_object_begin(_object, _set)
#define LINKER_SET_OBJECT_LIMIT(_object, _set)  __linker_set_object_limit(_object, _set)

#define LINKER_SET_OBJECT_FOREACH(_object, _pvar, _cast, _set)          \
	for (_pvar = (_cast) LINKER_SET_OBJECT_BEGIN(_object, _set);    \
	     _pvar < (_cast) LINKER_SET_OBJECT_LIMIT(_object, _set);    \
	     _pvar++)

#define LINKER_SET_OBJECT_ITEM(_object, _cast, _set, _i)                \
	(((_cast)(LINKER_SET_OBJECT_BEGIN(_object, _set)))[_i])

#define LINKER_SET_FOREACH(_pvar, _cast, _set)                                  \
	LINKER_SET_OBJECT_FOREACH((MACH_HEADER_TYPE *)MH_EXECUTE_HEADER, _pvar, _cast, _set)

/*
 * Implementation.
 *
 * void **__linker_set_object_begin(_header, _set)
 *	Returns a pointer to the first pointer in the linker set.
 * void **__linker_set_object_limi(_header, _set)
 *	Returns an upper bound to the linker set (base + size).
 */

static __inline intptr_t
__linker_get_slide(struct mach_header *_header)
{
#ifndef KERNEL
	return _dyld_get_image_slide(_header);
#else
	(void)_header;
	return 0;
#endif
}

static __inline void **
__linker_set_object_begin(MACH_HEADER_TYPE *_header, const char *_set)
__attribute__((__const__));
static __inline void **
__linker_set_object_begin(MACH_HEADER_TYPE *_header, const char *_set)
{
	char *_set_begin;
	SECTDATA_SIZE_TYPE _size;

	_set_begin = (char *)GETSECTIONDATA_VARIANT(_header, LINKER_SET_SEGMENT_CSTR, _set, &_size);
	_set_begin += __linker_get_slide((struct mach_header *)_header);
	return (void **)(uintptr_t)_set_begin;
}

static __inline void **
__linker_set_object_limit(MACH_HEADER_TYPE *_header, const char *_set)
__attribute__((__const__));
static __inline void **
__linker_set_object_limit(MACH_HEADER_TYPE *_header, const char *_set)
{
	char *_set_begin;
	SECTDATA_SIZE_TYPE _size;

	_set_begin = (char *)GETSECTIONDATA_VARIANT(_header, LINKER_SET_SEGMENT_CSTR, _set, &_size);
	_set_begin += __linker_get_slide((struct mach_header *)_header);

	return (void **) ((uintptr_t) _set_begin + _size);
}

#endif /* !KERNEL || __APPLE_API_PRIVATE */

#endif /* _SYS_LINKER_SET_H_ */
