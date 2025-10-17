/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, April 22, 2024.
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
#ifdef __MWERKS__
#define _CPP_ATTACHFACTORY
#endif
#include "attachfactory.h"

#include "cspattachment.h"
#include <Security/cssmdli.h>
#include <Security/cssmcli.h>
#include <Security/cssmaci.h>
#include <Security/cssmtpi.h>
#include <derived_src/funcnames.gen>
#include <map>


//
// A template to generate AttachmentMakers for the standard plugin types.
//
template <CSSM_SERVICE_TYPE type, typename Table, const char *const nameTable[]>
class StandardAttachmentMaker : public AttachmentMaker {
public:
    StandardAttachmentMaker() : AttachmentMaker(type)
    {
        for (unsigned n = 0; n < sizeof(nameTable) / sizeof(nameTable[0]); n++)
            nameMap.insert(typename NameMap::value_type(nameTable[n], n));
    }

    Attachment *make(Module *module,
                     const CSSM_VERSION &version,
                     uint32 subserviceId,
                     CSSM_SERVICE_TYPE subserviceType,
                     const CSSM_API_MEMORY_FUNCS &memoryOps,
                     CSSM_ATTACH_FLAGS attachFlags,
                     CSSM_KEY_HIERARCHY keyHierarchy,
                     CSSM_FUNC_NAME_ADDR *FunctionTable,
                     uint32 NumFunctions)
    {
        StandardAttachment<type, Table> *attachment =
        new StandardAttachment<type, Table>(module,
                                             nameMap,
                                             version,
                                             subserviceId,
                                             subserviceType,
                                             memoryOps,
                                             attachFlags,
                                             keyHierarchy);
        attachment->resolveSymbols(FunctionTable, NumFunctions);
        return attachment;
    }

private:
    typedef typename StandardAttachment<type, Table>::NameMap NameMap;
    NameMap nameMap;
};


//
// Implementation of an attachment factory
//
AttachmentFactory::AttachmentFactory()
{
    // generate explicitly known attachment types
    factories[CSSM_SERVICE_AC] = new StandardAttachmentMaker<CSSM_SERVICE_AC, CSSM_SPI_AC_FUNCS, ACNameTable>;
    factories[CSSM_SERVICE_CL] = new StandardAttachmentMaker<CSSM_SERVICE_CL, CSSM_SPI_CL_FUNCS, CLNameTable>;
    factories[CSSM_SERVICE_CSP] = new StandardAttachmentMaker<CSSM_SERVICE_CSP, CSSM_SPI_CSP_FUNCS, CSPNameTable>;
    factories[CSSM_SERVICE_DL] = new StandardAttachmentMaker<CSSM_SERVICE_DL, CSSM_SPI_DL_FUNCS, DLNameTable>;
    factories[CSSM_SERVICE_TP] = new StandardAttachmentMaker<CSSM_SERVICE_TP, CSSM_SPI_TP_FUNCS, TPNameTable>;
}


AttachmentMaker *AttachmentFactory::attachmentMakerFor(CSSM_SERVICE_TYPE type) const
{
    AttachFactories::const_iterator it = factories.find(type);
    if (it == factories.end())
        CssmError::throwMe(CSSMERR_CSSM_INVALID_SERVICE_MASK);
    return it->second;
}


//
// Manage an AttachmentMaker
//
AttachmentMaker::~AttachmentMaker()
{
}
