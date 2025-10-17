/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Monday, January 24, 2022.
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

#ifndef __APPLEHELPERS_H__
#define __APPLEHELPERS_H__ 1

#define __COREFOUNDATION_CFFILESECURITY__
#include <CoreFoundation/CoreFoundation.h>
#include <AccountPolicy/AccountPolicy.h>

#define kDisabledByAdmin               1
#define kDisabledExpired               2
#define kDisabledInactive              3
#define kDisabledTooManyFailedLogins   4
#define kDisabledNewPasswordRequired   5

Attribute *odusers_copy_passwordRequiredDate(void);
Attribute  *odusers_copy_attr(char *dn, char*attribute);
Entry *odusers_copy_entry(Operation *op);
int odusers_remove_authdata(char *slotid);
int odusers_get_authguid(Entry *e, char *guidstr);
Entry *odusers_copy_authdata(struct berval *dn);
Attribute *odusers_copy_globalpolicy(void);
CFDictionaryRef odusers_copy_effectiveuserpoldict(struct berval *dn);
CFDictionaryRef odusers_copy_defaultglobalpolicy(void);
int odusers_isdisabled(CFDictionaryRef policy);
bool odusers_isadmin(CFDictionaryRef policy);
struct berval *odusers_copy_dict2bv(CFDictionaryRef dict);
int odusers_increment_failedlogin(struct berval *dn);
int odusers_reset_failedlogin(struct berval *dn);
int odusers_successful_auth(struct berval *dn, CFDictionaryRef policy);
char *odusers_copy_saslrealm(void);
char *odusers_copy_suffix(void);
char *odusers_copy_krbrealm(Operation *op);
char *odusers_copy_recname(Operation *op);
char *odusers_copy_primarymasterip(Operation *op);
int odusers_set_password(struct berval *dn, char *password, int isChangingOwnPassword);
CFMutableDictionaryRef CopyPolicyToDict(const char *policyplist, int len);
int odusers_verify_passwordquality(const char *password, const char *username, CFDictionaryRef effectivepolicy, SlapReply *rs);
CFArrayRef odusers_copy_enabledmechs(const char *suffix);
int odusers_krb_auth(Operation *op, char *password);
char *odusers_copy_owner(struct berval *dn);
int odusers_store_history(struct berval *dn, const char *password);
char *CopyPrimaryIPv4Address(void);
bool odusers_ismember(struct berval *userdn, struct berval *groupdn);
int odusers_joingroup(const char *group, struct berval *dn, bool remove);

CFDictionaryRef odusers_copy_accountpolicy_fromentry(Entry *authe);
CFDictionaryRef  odusers_copy_globalaccountpolicy();
CFStringRef  odusers_copy_globalaccountpolicyGUID();
int odusers_accountpolicy_set(struct berval *dn, Entry *authe, CFDictionaryRef accountpolicydict);
CFDictionaryRef odusers_copy_accountpolicy(struct berval *dn);
CFDictionaryRef odusers_copy_accountpolicyinfo(struct berval *dn);
void odusers_accountpolicy_set_passwordinfo(CFMutableDictionaryRef accountpolicyinfo, const char *password);
CFDictionaryRef odusers_accountpolicy_retrievedata( CFDictionaryRef *policyData, CFArrayRef keys, struct berval *dn );
void odusers_accountpolicy_updatedata( CFDictionaryRef keys, struct berval *dn );
int odusers_accountpolicy_successful_auth(struct berval *dn, CFDictionaryRef policy);
int odusers_accountpolicy_failed_auth(struct berval *dn, CFDictionaryRef policy);
int odusers_accountpolicy_override(struct berval *account_dn);

#endif /* __APPLEHELPERS_H__ */
