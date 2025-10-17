/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Wednesday, March 15, 2023.
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
//
//
// WARNING! HERE BE DRAGONS!
// This code involves moderately arcane magic including (but not limited to)
// dancing macros paired off with self-maintaining stack objects. Don't take
// anything for granted! Be very afraid of ALL-CAPS names. Your best bet is
// probably to stick with the existing patterns.
//
#ifndef _H_TDTRANSIT
#define _H_TDTRANSIT

#include <security_tokend_client/tdclient.h>
#include <security_cdsa_utilities/cssmwalkers.h>
#include <SecurityTokend/SecTokend.h>
#include "tokend_types.h"
#include "tokend.h"

namespace Security {
namespace Tokend {


// stock leading argument profile used by all calls
#define TOKEND_ARGS mServicePort, mReplyPort, &rcode

// IPC wraps the actual MIG call
#define IPC(statement) \
	{ CSSM_RETURN rcode; check(statement); if (rcode != CSSM_OK) CssmError::throwMe(rcode); }

// pass mandatory or optional CssmData arguments into an IPC call
#define DATA(arg)			arg.data(), arg.length()
#define OPTIONALDATA(arg)	(arg ? arg->data() : NULL), (arg ? arg->length() : 0)

// pass structured arguments in/out of IPC calls. See "data walkers" for details
#define COPY(copy)			copy, copy.length(), copy
#define COPYFLAT(copy)		copy, copy##Length, copy
#define COPY_OUT(copy)		&copy, &copy##Length, &copy##Base
#define COPY_OUT_DECL(type,name) type *name, *name##Base; mach_msg_type_number_t name##Length


//
// DataOutput manages an output CssmData argument.
//
class DataOutput {
public:
	DataOutput(CssmData &arg, Allocator &alloc)
		: argument(arg), allocator(alloc) { mData = NULL; mLength = 0; }
	~DataOutput();
	
	void **data() { return &mData; }
	mach_msg_type_number_t *length() { return &mLength; }
	
	CssmData &argument;
	Allocator &allocator;

private:
	void *mData;
	mach_msg_type_number_t mLength;
};


//
// Bundle up a Context for IPC transmission
//
class SendContext {
public:
	SendContext(const Context &ctx);
	~SendContext() { Allocator::standard().free(attributes); }
	
	const Context &context;
	CSSM_CONTEXT_ATTRIBUTE *attributes;
	size_t attributeSize;
};

#define CONTEXT(ctx)	ctx.context, ctx.attributes, ctx.attributes, ctx.attributeSize


//
// A PodWrapper for TOKEND_RETURN_DATA (used in the tokend APIs)
//
class TokendReturnData : public PodWrapper<TokendReturnData, TOKEND_RETURN_DATA> {
public:
};


}	// namespace Tokend
}	// namespace Security

#endif //_H_TDTRANSIT
