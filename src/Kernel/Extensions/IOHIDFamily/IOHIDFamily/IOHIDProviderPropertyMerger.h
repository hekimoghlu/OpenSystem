/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 31, 2021.
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
#include <IOKit/IOService.h>
#include <IOKit/IOProviderPropertyMerger.h>

#ifndef _IOKIT_IOHIDPROVIDERPROPERTYMERGER_H
#define _IOKIT_IOHIDPROVIDERPROPERTYMERGER_H
class IOHIDProviderPropertyMerger : public IOProviderPropertyMerger
{
    OSDeclareDefaultStructors(IOHIDProviderPropertyMerger);

protected:
    struct ExpansionData { };
    
    /*! @var reserved
        Reserved for future use.  (Internal use only)  */
    ExpansionData *reserved;

    virtual bool mergeProperties(IOService *  provider, OSDictionary * properties);
    virtual bool mergeDictionaries(OSDictionary * source, OSDictionary * target);

public:
    
    virtual IOService * probe(IOService * provider, SInt32 * score) APPLE_KEXT_OVERRIDE;
};

#endif /* ! _IOKIT_IOHIDPROVIDERPROPERTYMERGER_H */
