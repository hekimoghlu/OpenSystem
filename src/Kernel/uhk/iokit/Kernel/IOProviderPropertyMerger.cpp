/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, January 5, 2023.
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
#include <IOKit/IOProviderPropertyMerger.h>
#include <IOKit/IOService.h>

#define super IOService
OSDefineMetaClassAndStructors(IOProviderPropertyMerger, IOService);

bool
IOProviderPropertyMerger::init(OSDictionary * dictionary)
{
	OSDictionary *mergeProperties = OSDynamicCast(OSDictionary, dictionary->getObject(kIOProviderMergePropertiesKey));
	OSDictionary *parentMergeProperties = OSDynamicCast(OSDictionary, dictionary->getObject(kIOProviderParentMergePropertiesKey));

	// remove security-sensitive properties from the dictionary used to merge properties to provider
	if (mergeProperties) {
		mergeProperties->removeObject(gIOServiceDEXTEntitlementsKey);
	}
	if (parentMergeProperties) {
		parentMergeProperties->removeObject(gIOServiceDEXTEntitlementsKey);
	}

	return super::init(dictionary);
}

bool
IOProviderPropertyMerger::setProperty(const OSSymbol * aKey, OSObject * anObject)
{
	// Disallow modifying security-sensitive properties
	if (aKey->isEqualTo(kIOProviderMergePropertiesKey) || aKey->isEqualTo(kIOProviderParentMergePropertiesKey)) {
		return false;
	}
	return super::setProperty(aKey, anObject);
}

void
IOProviderPropertyMerger::setPropertyTable(OSDictionary * dict __unused)
{
	// Disallow changing the entire property table since that can change security-sensitive properties
}
