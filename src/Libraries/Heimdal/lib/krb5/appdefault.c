/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, March 2, 2025.
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
#include "krb5_locl.h"

KRB5_LIB_FUNCTION void KRB5_LIB_CALL
krb5_appdefault_boolean(krb5_context context, const char *appname,
			krb5_const_realm realm, const char *option,
			krb5_boolean def_val, krb5_boolean *ret_val)
{

    if(appname == NULL)
	appname = getprogname();

    def_val = krb5_config_get_bool_default(context, NULL, def_val,
					   "libdefaults", option, NULL);
    if(realm != NULL)
	def_val = krb5_config_get_bool_default(context, NULL, def_val,
					       "realms", realm, option, NULL);

    def_val = krb5_config_get_bool_default(context, NULL, def_val,
					   "appdefaults",
					   option,
					   NULL);
    if(realm != NULL)
	def_val = krb5_config_get_bool_default(context, NULL, def_val,
					       "appdefaults",
					       realm,
					       option,
					       NULL);
    if(appname != NULL) {
	def_val = krb5_config_get_bool_default(context, NULL, def_val,
					       "appdefaults",
					       appname,
					       option,
					       NULL);
	if(realm != NULL)
	    def_val = krb5_config_get_bool_default(context, NULL, def_val,
						   "appdefaults",
						   appname,
						   realm,
						   option,
						   NULL);
    }
    *ret_val = def_val;
}

KRB5_LIB_FUNCTION void KRB5_LIB_CALL
krb5_appdefault_string(krb5_context context, const char *appname,
		       krb5_const_realm realm, const char *option,
		       const char *def_val, char **ret_val)
{
    if(appname == NULL)
	appname = getprogname();

    def_val = krb5_config_get_string_default(context, NULL, def_val,
					     "libdefaults", option, NULL);
    if(realm != NULL)
	def_val = krb5_config_get_string_default(context, NULL, def_val,
						 "realms", realm, option, NULL);

    def_val = krb5_config_get_string_default(context, NULL, def_val,
					     "appdefaults",
					     option,
					     NULL);
    if(realm != NULL)
	def_val = krb5_config_get_string_default(context, NULL, def_val,
						 "appdefaults",
						 realm,
						 option,
						 NULL);
    if(appname != NULL) {
	def_val = krb5_config_get_string_default(context, NULL, def_val,
						 "appdefaults",
						 appname,
						 option,
						 NULL);
	if(realm != NULL)
	    def_val = krb5_config_get_string_default(context, NULL, def_val,
						     "appdefaults",
						     appname,
						     realm,
						     option,
						     NULL);
    }
    if(def_val != NULL)
	*ret_val = strdup(def_val);
    else
	*ret_val = NULL;
}

KRB5_LIB_FUNCTION void KRB5_LIB_CALL
krb5_appdefault_time(krb5_context context, const char *appname,
		     krb5_const_realm realm, const char *option,
		     time_t def_val, time_t *ret_val)
{
    krb5_deltat t;
    char *val;

    krb5_appdefault_string(context, appname, realm, option, NULL, &val);
    if (val == NULL) {
	*ret_val = def_val;
	return;
    }
    if (krb5_string_to_deltat(val, &t))
	*ret_val = def_val;
    else
	*ret_val = t;
    free(val);
}
