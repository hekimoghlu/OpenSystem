/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Friday, August 1, 2025.
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

#include "NullTransform.h"

NullTransform::NullTransform() : Transform(CFSTR("NullTransform"))
{
}



CFTypeRef NullTransform::Make() CF_RETURNS_RETAINED
{
	return CoreFoundationHolder::MakeHolder(gInternalCFObjectName, new NullTransform());
}



void NullTransform::AttributeChanged(CFStringRef name, CFTypeRef value)
{
	// move input to output, otherwise do nothing
	if (CFStringCompare(name, kSecTransformInputAttributeName, 0) == kCFCompareEqualTo)
	{
		SetAttributeNoCallback(kSecTransformOutputAttributeName, value);
	}
}



std::string NullTransform::DebugDescription()
{
	return Transform::DebugDescription() + ": NullTransform";
}



class NullTransformFactory : public TransformFactory
{
public:
	NullTransformFactory();
	
	virtual CFTypeRef Make();
};



TransformFactory* NullTransform::MakeTransformFactory()
{
	return new NullTransformFactory();
}



NullTransformFactory::NullTransformFactory() : TransformFactory(CFSTR("Null Transform"))
{
}



CFTypeRef NullTransformFactory::Make()
{
	return NullTransform::Make();
}

