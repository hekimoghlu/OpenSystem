/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 20, 2022.
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
#ifndef	_SEC_ASN1_CODER_H_
#define _SEC_ASN1_CODER_H_

#include <sys/types.h>
#include <Security/SecAsn1Types.h>
#include <TargetConditionals.h>
#include <Security/SecBase.h> /* error codes */

#ifdef __cplusplus
extern "C" {
#endif

CF_ASSUME_NONNULL_BEGIN

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"

/*
 * Opaque reference to a SecAsn1Coder object.
 */
typedef struct SecAsn1Coder *SecAsn1CoderRef SEC_ASN1_API_DEPRECATED;

/*
 * Create/destroy SecAsn1Coder object. 
 */
OSStatus SecAsn1CoderCreate(
	SecAsn1CoderRef  __nullable * __nonnull coder) SEC_ASN1_API_DEPRECATED;

OSStatus SecAsn1CoderRelease(
	SecAsn1CoderRef  coder) SEC_ASN1_API_DEPRECATED;

/*
 * DER decode an untyped item per the specified template array. 
 * The result is allocated in this SecAsn1Coder's memory pool and 
 * is freed when this object is released.
 *
 * The templates argument points to a an array of SecAsn1Templates 
 * defining the object to be decoded; the end of the array is 
 * indicated by a SecAsn1Template with file kind equalling 0. 
 *
 * The dest pointer is a template-specific struct allocated by the caller 
 * and must be zeroed by the caller. 
 *
 * Returns errSecUnknownFormat on decode-specific error.
 */
OSStatus SecAsn1Decode(
	SecAsn1CoderRef			coder,
	const void				*src,		// DER-encoded source
	size_t					len,
	const SecAsn1Template 	*templates,	
	void					*dest) SEC_ASN1_API_DEPRECATED;

/* 
 * Convenience routine, decode from a SecAsn1Item.
 */
OSStatus SecAsn1DecodeData(
	SecAsn1CoderRef			coder,
	const SecAsn1Item		*src,
	const SecAsn1Template 	*templ,	
	void					*dest) SEC_ASN1_API_DEPRECATED;

/*
 * DER encode. The encoded data (in dest.Data) is allocated in this 
 * SecAsn1Coder's memory pool and is freed when this object is released.
 *
 * The src pointer is a template-specific struct.
 *
 * The templates argument points to a an array of SecAsn1Templates 
 * defining the object to be decoded; the end of the array is 
 * indicated by a SecAsn1Template with file kind equalling 0. 
 */
OSStatus SecAsn1EncodeItem(
	SecAsn1CoderRef			coder,
	const void				*src,
	const SecAsn1Template 	*templates,	
	SecAsn1Item				*dest) SEC_ASN1_API_DEPRECATED;

/*
 * Some alloc-related methods which come in handy when using
 * this object. All memory is allocated using this object's 
 * memory pool. Caller never has to free it. Used for
 * temp allocs of memory which only needs a scope which is the
 * same as this object. 
 *
 * All except SecAsn1Malloc return a errSecAllocate in the highly 
 * unlikely event of a malloc failure.
 *
 * SecAsn1Malloc() returns a pointer to allocated memory, like 
 * malloc().
 */
void *SecAsn1Malloc(
	SecAsn1CoderRef			coder,
	size_t					len) SEC_ASN1_API_DEPRECATED;

/* Allocate item.Data, set item.Length */
OSStatus SecAsn1AllocItem(
	SecAsn1CoderRef			coder,
	SecAsn1Item				*item,
	size_t					len) SEC_ASN1_API_DEPRECATED;

/* Allocate and copy, various forms */
OSStatus SecAsn1AllocCopy(
	SecAsn1CoderRef			coder,
	const void				*src,		/* memory copied from here */
	size_t					len,		/* length to allocate & copy */
	SecAsn1Item				*dest)		/* dest->Data allocated and copied to;
										 *   dest->Length := len */
    SEC_ASN1_API_DEPRECATED;

OSStatus SecAsn1AllocCopyItem(
	SecAsn1CoderRef			coder,
	const SecAsn1Item		*src,		/* src->Length bytes allocated and copied from
										 *   src->Data */
	SecAsn1Item				*dest)		/* dest->Data allocated and copied to;
										 *   dest->Length := src->Length */
    SEC_ASN1_API_DEPRECATED;

/* Compare two decoded OIDs.  Returns true iff they are equivalent. */
bool SecAsn1OidCompare(const SecAsn1Oid *oid1, const SecAsn1Oid *oid2) SEC_ASN1_API_DEPRECATED;

#pragma clang diagnostic pop

CF_ASSUME_NONNULL_END

#ifdef __cplusplus
}
#endif

#endif	/* _SEC_ASN1_CODER_H_ */
