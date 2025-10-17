/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 27, 2024.
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
#include <ffi.h>
#include <ffi_common.h>

#include <stdbool.h>
#include <stdlib.h>

/* Round up to FFI_SIZEOF_ARG. */
#define STACK_ARG_SIZE(x) ALIGN(x, FFI_SIZEOF_ARG)

/* Perform machine independent initialization of aggregate type
   specifications. */

static ffi_status
initialize_aggregate(
/*@out@*/	ffi_type*	arg)
{
/*@-usedef@*/

	if (arg == NULL || arg->elements == NULL ||
		arg->size != 0 || arg->alignment != 0)
		return FFI_BAD_TYPEDEF;

	ffi_type**	ptr = &(arg->elements[0]);

	while ((*ptr) != NULL)
	{
		if (((*ptr)->size == 0) && (initialize_aggregate(*ptr) != FFI_OK))
			return FFI_BAD_TYPEDEF;

		/* Perform a sanity check on the argument type */
		FFI_ASSERT_VALID_TYPE(*ptr);

#ifdef POWERPC_DARWIN
		int curalign = (*ptr)->alignment;

		if (ptr != &(arg->elements[0]))
		{
			if (curalign > 4 && curalign != 16)
				curalign = 4;
		}

		arg->size		= ALIGN(arg->size, curalign);
		arg->size		+= (*ptr)->size;
		arg->alignment	= (arg->alignment > curalign) ? 
			arg->alignment : curalign;
#else
		arg->size		= ALIGN(arg->size, (*ptr)->alignment);
		arg->size		+= (*ptr)->size;
		arg->alignment	= (arg->alignment > (*ptr)->alignment) ? 
			arg->alignment : (*ptr)->alignment;
#endif

		ptr++;
    }

  /* Structure size includes tail padding.  This is important for
     structures that fit in one register on ABIs like the PowerPC64
     Linux ABI that right justify small structs in a register.
     It's also needed for nested structure layout, for example
     struct A { long a; char b; }; struct B { struct A x; char y; };
     should find y at an offset of 2*sizeof(long) and result in a
     total size of 3*sizeof(long).  */
	arg->size = ALIGN(arg->size, arg->alignment);

	if (arg->size == 0)
		return FFI_BAD_TYPEDEF;

	return FFI_OK;

/*@=usedef@*/
}

#ifndef __CRIS__
/* The CRIS ABI specifies structure elements to have byte
   alignment only, so it completely overrides this functions,
   which assumes "natural" alignment and padding.  */

/* Perform machine independent ffi_cif preparation, then call
   machine dependent routine. */

#if defined(X86_DARWIN)

static inline bool
struct_on_stack(
	int	size)
{
	if (size > 8)
		return true;

	/* This is not what the ABI says, but is what is really implemented */
	switch (size)
	{
		case 1:
		case 2:
		case 4:
		case 8:
			return false;

		default:
			return true;
	}
}

#endif	// defined(X86_DARWIN)

// Arguments' ffi_type->alignment must be nonzero.
ffi_status
ffi_prep_cif(
/*@out@*/ /*@partial@*/	ffi_cif*		cif, 
						ffi_abi			abi,
						unsigned int	nargs, 
/*@dependent@*/ /*@out@*/ /*@partial@*/ ffi_type*	rtype, 
/*@dependent@*/			ffi_type**		atypes)
{
	if (cif == NULL)
		return FFI_BAD_TYPEDEF;

	if (abi <= FFI_FIRST_ABI || abi > FFI_DEFAULT_ABI)
		return FFI_BAD_ABI;

	unsigned int	bytes	= 0;
	unsigned int	i;
	ffi_type**		ptr;

	cif->abi = abi;
	cif->arg_types = atypes;
	cif->nargs = nargs;
	cif->rtype = rtype;
	cif->flags = 0;

	/* Initialize the return type if necessary */
	/*@-usedef@*/
	if ((cif->rtype->size == 0) && (initialize_aggregate(cif->rtype) != FFI_OK))
		return FFI_BAD_TYPEDEF;
	/*@=usedef@*/

	/* Perform a sanity check on the return type */
	FFI_ASSERT_VALID_TYPE(cif->rtype);

	/* x86-64 and s390 stack space allocation is handled in prep_machdep.  */
#if !defined M68K && !defined __x86_64__ && !defined S390 && !defined PA
	/* Make space for the return structure pointer */
	if (cif->rtype->type == FFI_TYPE_STRUCT
#ifdef SPARC
		&& (cif->abi != FFI_V9 || cif->rtype->size > 32)
#endif
#ifdef X86_DARWIN
		&& (struct_on_stack(cif->rtype->size))
#endif
		)
		bytes = STACK_ARG_SIZE(sizeof(void*));
#endif

	for (ptr = cif->arg_types, i = cif->nargs; i > 0; i--, ptr++)
	{
		/* Initialize any uninitialized aggregate type definitions */
		if (((*ptr)->size == 0) && (initialize_aggregate((*ptr)) != FFI_OK))
			return FFI_BAD_TYPEDEF;

		if ((*ptr)->alignment == 0)
			return FFI_BAD_TYPEDEF;

		/* Perform a sanity check on the argument type, do this 
		check after the initialization.  */
		FFI_ASSERT_VALID_TYPE(*ptr);

#if defined(X86_DARWIN)
		{
			int align = (*ptr)->alignment;

			if (align > 4)
				align = 4;

			if ((align - 1) & bytes)
				bytes = ALIGN(bytes, align);

			bytes += STACK_ARG_SIZE((*ptr)->size);
		}
#elif !defined __x86_64__ && !defined S390 && !defined PA
#ifdef SPARC
		if (((*ptr)->type == FFI_TYPE_STRUCT
			&& ((*ptr)->size > 16 || cif->abi != FFI_V9))
			|| ((*ptr)->type == FFI_TYPE_LONGDOUBLE
			&& cif->abi != FFI_V9))
				bytes += sizeof(void*);
		else
#endif
		{
			/* Add any padding if necessary */
			if (((*ptr)->alignment - 1) & bytes)
				bytes = ALIGN(bytes, (*ptr)->alignment);

			bytes += STACK_ARG_SIZE((*ptr)->size);
		}
#endif
	}

	cif->bytes = bytes;

	/* Perform machine dependent cif processing */
	return ffi_prep_cif_machdep(cif);
}
#endif /* not __CRIS__ */
