/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, October 22, 2023.
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
#include <security_cdsa_plugin/ACsession.h>
#include <security_cdsa_plugin/cssmplugin.h>
#include <security_cdsa_utilities/cssmbridge.h>
#include <Security/cssmaci.h>
// The disclaimer up top used to be true but rdar://24426719 removed libsecurity_cdsa_plugin_generate
#include "LegacyAPICounts.h"

ACAbstractPluginSession::~ACAbstractPluginSession()
{ /* virtual */ }

static CSSM_RETURN CSSMACI cssm_AuthCompute(CSSM_AC_HANDLE ACHandle,
         const CSSM_TUPLEGROUP *BaseAuthorizations,
         const CSSM_TUPLEGROUP *Credentials,
         uint32 NumberOfRequestors,
         const CSSM_LIST *Requestors,
         const CSSM_LIST *RequestedAuthorizationPeriod,
         const CSSM_LIST *RequestedAuthorization,
         CSSM_TUPLEGROUP_PTR AuthorizationResult)
{
  BEGIN_API
  findSession<ACPluginSession>(ACHandle).AuthCompute(Required(BaseAuthorizations),
			Credentials,
			NumberOfRequestors,
			Required(Requestors),
			RequestedAuthorizationPeriod,
			Required(RequestedAuthorization),
			Required(AuthorizationResult));
  END_API(AC)
}

static CSSM_RETURN CSSMACI cssm_PassThrough(CSSM_AC_HANDLE ACHandle,
         CSSM_TP_HANDLE TPHandle,
         CSSM_CL_HANDLE CLHandle,
         CSSM_CC_HANDLE CCHandle,
         const CSSM_DL_DB_LIST *DBList,
         uint32 PassThroughId,
         const void *InputParams,
         void **OutputParams)
{
  BEGIN_API
  findSession<ACPluginSession>(ACHandle).PassThrough(TPHandle,
			CLHandle,
			CCHandle,
			Required(DBList),
			PassThroughId,
			InputParams,
			OutputParams);
  END_API(AC)
}


static const CSSM_SPI_AC_FUNCS ACFunctionStruct = {
  cssm_AuthCompute,
  cssm_PassThrough,
};

static CSSM_MODULE_FUNCS ACFunctionTable = {
  CSSM_SERVICE_AC,	// service type
  2,	// number of functions
  (const CSSM_PROC_ADDR *)&ACFunctionStruct
};

CSSM_MODULE_FUNCS_PTR ACPluginSession::construct()
{
   return &ACFunctionTable;
}
