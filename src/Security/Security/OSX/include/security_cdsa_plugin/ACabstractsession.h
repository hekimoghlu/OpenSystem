/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, February 5, 2023.
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
// AC plugin transition layer.
// This file was automatically generated. Do not edit on penalty of futility!
//
#ifndef _H_ACABSTRACTSESSION
#define _H_ACABSTRACTSESSION

#include <security_cdsa_plugin/pluginsession.h>
#include <security_cdsa_utilities/cssmdata.h>


namespace Security {


//
// A pure abstract class to define the AC module interface
//
class ACAbstractPluginSession {
public:
	virtual ~ACAbstractPluginSession();
  virtual void AuthCompute(const CSSM_TUPLEGROUP &BaseAuthorizations,
         const CSSM_TUPLEGROUP *Credentials,
         uint32 NumberOfRequestors,
         const CSSM_LIST &Requestors,
         const CSSM_LIST *RequestedAuthorizationPeriod,
         const CSSM_LIST &RequestedAuthorization,
         CSSM_TUPLEGROUP &AuthorizationResult) = 0;
  virtual void PassThrough(CSSM_TP_HANDLE TPHandle,
         CSSM_CL_HANDLE CLHandle,
         CSSM_CC_HANDLE CCHandle,
         const CSSM_DL_DB_LIST &DBList,
         uint32 PassThroughId,
         const void *InputParams,
         void **OutputParams) = 0;
};

} // end namespace Security

#endif //_H_ACABSTRACTSESSION
