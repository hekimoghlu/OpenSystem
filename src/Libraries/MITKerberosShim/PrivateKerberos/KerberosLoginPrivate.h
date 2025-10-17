/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, August 25, 2025.
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
#ifndef __KERBEROSLOGINPRIVATE__
#define __KERBEROSLOGINPRIVATE__

#if defined(macintosh) || (defined(__MACH__) && defined(__APPLE__))
#    include <TargetConditionals.h>
#    if TARGET_RT_MAC_CFM
#        error "Use KfM 4.0 SDK headers for CFM compilation."
#    endif
#endif

#include <Kerberos/KerberosLogin.h>
#include <Kerberos/krb5.h>

#ifdef __cplusplus
extern "C" {
#endif

enum {
    klPromptMechanism_Autodetect = 0,
    klPromptMechanism_GUI = 1,
    klPromptMechanism_CLI = 2,
    klPromptMechanism_None = 0xFFFFFFFF
};
typedef uint32_t KLPromptMechanism;

/*************/
/*** Types ***/
/*************/

#ifdef KERBEROSLOGIN_DEPRECATED

typedef krb5_error_code (*KLPrompterProcPtr) (krb5_context  context,
                                              void         *data,
                                              const char   *name,
                                              const char   *banner,
                                              int           num_prompts,
                                              krb5_prompt   prompts[]);
KLStatus __KLSetApplicationPrompter (KLPrompterProcPtr inPrompter);

#endif /* KERBEROSLOGIN_DEPRECATED */
    
/*****************/
/*** Functions ***/
/*****************/

KLStatus  __KLSetHomeDirectoryAccess (KLBoolean inAllowHomeDirectoryAccess);
KLBoolean __KLAllowHomeDirectoryAccess (void);

KLStatus  __KLSetAutomaticPrompting (KLBoolean inAllowAutomaticPrompting);
KLBoolean __KLAllowAutomaticPrompting (void);

KLBoolean __KLAllowRememberPassword (void);

KLStatus          __KLSetPromptMechanism (KLPromptMechanism inPromptMechanism);
KLPromptMechanism __KLPromptMechanism (void);

KLStatus __KLCreatePrincipalFromTriplet (const char  *inName,
                                         const char  *inInstance,
                                         const char  *inRealm,
                                         KLKerberosVersion  inKerberosVersion,
                                         KLPrincipal *outPrincipal);

KLStatus __KLGetTripletFromPrincipal (KLPrincipal         inPrincipal,
                                      KLKerberosVersion   inKerberosVersion,
                                      char              **outName,
                                      char              **outInstance,
                                      char              **outRealm);

KLStatus __KLCreatePrincipalFromKerberos5Principal (krb5_principal  inPrincipal,
                                                    KLPrincipal    *outPrincipal);

KLStatus __KLGetKerberos5PrincipalFromPrincipal (KLPrincipal     inPrincipal, 
                                                 krb5_context    inContext, 
                                                 krb5_principal *outKrb5Principal);

KLBoolean __KLPrincipalIsTicketGrantingService (KLPrincipal inPrincipal);

KLStatus __KLGetKeychainPasswordForPrincipal (KLPrincipal   inPrincipal,
                                              char        **outPassword);

KLStatus __KLPrincipalSetKeychainPassword (KLPrincipal  inPrincipal,
                                           const char  *inPassword);

KLStatus __KLRemoveKeychainPasswordForPrincipal (KLPrincipal inPrincipal);

#if TARGET_OS_MAC
#    if defined(__MWERKS__)
#        pragma import reset
#    endif
#endif

#ifdef __cplusplus
}
#endif

#endif /* __KERBEROSLOGINPRIVATE__ */

