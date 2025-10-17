/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, August 22, 2024.
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
// context - manage CSSM (cryptographic) contexts every which way.
//
// A note on memory management:
// Context attributes are allocated from application memory in big chunks comprising
// many attributes as well as the attribute array itself. The CSSM_CONTEXT fields
// NumberOfAttributes and ContextAttributes are handled as a group. Context::Builder
// and Context::copyFrom assume these fields are undefined and fill them. Context::clear
// assumes they are valid and invalides them, freeing memory.
//
#ifdef __MWERKS__
#define _CPP_CSSMCONTEXT
#endif
#include "cssmcontext.h"


//
// Destroy a HandleContext.
//
HandleContext::~HandleContext()
{
	attachment.free(extent);
	attachment.free(ContextAttributes);
}


//
// Locking protocol for HandleContexts
//
void HandleContext::lock()
{ attachment.enter(); }

bool HandleContext::tryLock()
{ return attachment.tryEnter(); }


//
// Merge a new set of attributes into an existing HandleContext, copying
// the new values deeply while releasing corresponding old values.
//
// NOTE: This is a HandleContext method; it does not work on bare Contexts.
//
void HandleContext::mergeAttributes(const CSSM_CONTEXT_ATTRIBUTE *attributes, uint32 count)
{
	// attempt to fast-path some simple or frequent cases
	if (count == 1) {
		if (Attr *attr = find(attributes[0].AttributeType)) {
			if (attr->baseType() == CSSM_ATTRIBUTE_DATA_UINT32) {
				// try quick replacement
				Attr oldAttr = *attr;
				*attr = attributes[0];
				if (CSSM_RETURN err = validateChange(CSSM_CONTEXT_EVENT_UPDATE)) {
					// roll back and fail
					*attr = oldAttr;
					CssmError::throwMe(err);
				}
				return;	// all done
			} else {
				// pointer value - does it fit into the space of the current value?
				size_t oldSize = size(*attr);
				size_t newSize = size(attributes[0]);
                Attr oldAttr = *attr;
				if (newSize <= oldSize) {	// give it a try...
					*attr = attributes[0];
					// NOTE that the CSP is getting a "temporary" pointer set to validate;
					// If we commit, the final copy will be elsewhere. CSP writer beware!
					if (CSSM_RETURN err = validateChange(CSSM_CONTEXT_EVENT_UPDATE)) {
						// roll back and fail
						*attr = oldAttr;
						CssmError::throwMe(err);
					}
					// commit new value
					CopyWalker copier(oldAttr.Attribute.String);
					walk(copier, *attr);
					return;
				}
			}
		} else {	// single change, new attribute
			if (Attr *slot = find(CSSM_ATTRIBUTE_NONE)) {
				const Attr *attr = static_cast<const Attr *>(&attributes[0]);
				if (attr->baseType() == CSSM_ATTRIBUTE_DATA_UINT32) {	// trivial
					Attr oldSlot = *slot;
					*slot = *attr;
					if (CSSM_RETURN err = validateChange(CSSM_CONTEXT_EVENT_UPDATE)) {
						*slot = oldSlot;
						CssmError::throwMe(err);
					}
					// already ok
					return;
				} else if (extent == NULL) {	// pointer value, allocate into extent
					void *data = attachment.malloc(size(*attr));
					try {
						Attr oldSlot = *slot;
						*slot = attributes[0];
						CopyWalker copier(data);
						walk(copier, *slot);
						if (CSSM_RETURN err = validateChange(CSSM_CONTEXT_EVENT_UPDATE)) {
							*slot = oldSlot;
							CssmError::throwMe(err);
						}
					} catch (...) {
						attachment.free(data);
						throw;
					}
					extent = data;
					return;
				}
			}
		}
	}
	
	// slow form: build a new value table and get rid of the old one
	Context::Builder builder(attachment);
	for (unsigned n = 0; n < count; n++)
		builder.setup(attributes[n]);
	for (unsigned n = 0; n < attributesInUse(); n++)
		if (!find(ContextAttributes[n].AttributeType, attributes, count))
			builder.setup(ContextAttributes[n]);
	builder.make();
	for (unsigned n = 0; n < count; n++)
		builder.put(attributes[n]);
	for (unsigned n = 0; n < attributesInUse(); n++)
		if (!find(ContextAttributes[n].AttributeType, attributes, count))
			builder.put(ContextAttributes[n]);
			
	// Carefully, now! The CSP may yet tell us to back out.
	// First, save the old values...
	CSSM_CONTEXT_ATTRIBUTE *oldAttributes = ContextAttributes;
	uint32 oldCount = NumberOfAttributes;
	
	// ...install new blob into the context...
	builder.done(ContextAttributes, NumberOfAttributes);
	
	// ...and ask the CSP whether this is okay
	if (CSSM_RETURN err = validateChange(CSSM_CONTEXT_EVENT_UPDATE)) {
		// CSP refused; put everything back where it belongs
		attachment.free(ContextAttributes);
		ContextAttributes = oldAttributes;
		NumberOfAttributes = oldCount;
		CssmError::throwMe(err);
	}
	
	// we succeeded, so NOW delete the old attributes blob
	attachment.free(oldAttributes);
}


//
// Ask the CSP to validate a proposed (and already implemented) change
//
CSSM_RETURN HandleContext::validateChange(CSSM_CONTEXT_EVENT event)
{
	// lock down the module if it is not thread-safe
	StLock<Module, &Module::safeLock, &Module::safeUnlock> _(attachment.module);
	return attachment.downcalls.EventNotify(attachment.handle(),
		event, handle(), this);
}


//
// Wrap up a deluxe context creation operation and return the new CC handle.
//
CSSM_CC_HANDLE HandleContext::Maker::operator () (CSSM_CONTEXT_TYPE type,
												CSSM_ALGORITHMS algorithm)
{
	// construct the HandleContext object
	HandleContext &context = *new(attachment) HandleContext(attachment, type, algorithm);
	context.CSPHandle = attachment.handle();
	done(context.ContextAttributes, context.NumberOfAttributes);
	
	// ask the CSP for consent
	if (CSSM_RETURN err = context.validateChange(CSSM_CONTEXT_EVENT_CREATE)) {
		// CSP refused; clean up and fail
		context.destroy(&context, context.attachment);
		CssmError::throwMe(err);
	}
	
	// return the new handle (we have succeeded)
	return context.handle();
}
