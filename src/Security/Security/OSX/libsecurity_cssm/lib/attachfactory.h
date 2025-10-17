/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, January 1, 2022.
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
// attachfactory - industrial grade production of Attachment objects
//
#ifndef _H_ATTACHFACTORY
#define _H_ATTACHFACTORY

#include "cssmint.h"
#include "attachment.h"

#ifdef _CPP_ATTACHFACTORY
# pragma export on
#endif


//
// An AttachmentMaker can create an Attachment object for a particular service
// type when asked nicely.
// 
class AttachmentMaker {
public:
    AttachmentMaker(CSSM_SERVICE_TYPE type) : mType(type) { }
    virtual ~AttachmentMaker();

    virtual Attachment *make(Module *module,
                             const CSSM_VERSION &version,
                             uint32 subserviceId,
                             CSSM_SERVICE_TYPE subserviceType,
                             const CSSM_API_MEMORY_FUNCS &memoryOps,
                             CSSM_ATTACH_FLAGS attachFlags,
                             CSSM_KEY_HIERARCHY keyHierarchy,
                             CSSM_FUNC_NAME_ADDR *functionTable,
                             uint32 functionTableSize) = 0;

    CSSM_SERVICE_TYPE factoryType() const { return mType; }
    
private:
    CSSM_SERVICE_TYPE mType;
};


//
// An AttachmentFactory contains a registry of AttachmentMakers for different
// service types, and produces the needed one on request.
//
class AttachmentFactory {
public:
    AttachmentFactory();
    
    AttachmentMaker *attachmentMakerFor(CSSM_SERVICE_TYPE type) const;

private:
    typedef map<CSSM_SERVICE_TYPE, AttachmentMaker *> AttachFactories;
    AttachFactories factories;
};

#ifdef _CPP_ATTACHFACTORY
# pragma export off
#endif


#endif //_H_ATTACHFACTORY
