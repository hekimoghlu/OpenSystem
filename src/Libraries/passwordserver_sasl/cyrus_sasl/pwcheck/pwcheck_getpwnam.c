/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, September 17, 2022.
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
#include <pwd.h>

extern char *crypt();

char *pwcheck(userid, password)
char *userid;
char *password;
{
    char* r;
    struct passwd *pwd;

    pwd = getpwnam(userid);
    if (!pwd) {
	r = "Userid not found";
    }
    else if (pwd->pw_passwd[0] == '*') {
	r = "Account disabled";
    }
    else if (strcmp(pwd->pw_passwd, crypt(password, pwd->pw_passwd)) != 0) {
	r = "Incorrect password";
    }
    else {
	r = "OK";
    }

    endpwent();
    
    return r;
}
