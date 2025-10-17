/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Sunday, November 19, 2023.
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
#include "includes.h"

#include <sys/types.h>
#include <string.h>
#include <unistd.h>
#include <pwd.h>

# if defined(HAVE_CRYPT_H) && !defined(HAVE_SECUREWARE)
#  include <crypt.h>
# endif

# ifdef __hpux
#  include <hpsecurity.h>
#  include <prot.h>
# endif

# ifdef HAVE_SECUREWARE
#  include <sys/security.h>
#  include <sys/audit.h>
#  include <prot.h>
# endif

# if defined(HAVE_SHADOW_H) && !defined(DISABLE_SHADOW)
#  include <shadow.h>
# endif

# if defined(HAVE_GETPWANAM) && !defined(DISABLE_SHADOW)
#  include <sys/label.h>
#  include <sys/audit.h>
#  include <pwdadj.h>
# endif

# if defined(WITH_OPENSSL) && !defined(HAVE_CRYPT) && defined(HAVE_DES_CRYPT)
#  include <openssl/des.h>
#  define crypt DES_crypt
# endif

#define MINIMUM(a, b)	(((a) < (b)) ? (a) : (b))

/*
 * Pick an appropriate password encryption type and salt for the running
 * system by searching through accounts until we find one that has a valid
 * salt.  Usually this will be root unless the root account is locked out.
 * If we don't find one we return a traditional DES-based salt.
 */
static const char *
pick_salt(void)
{
	struct passwd *pw;
	char *passwd, *p;
	size_t typelen;
	static char salt[32];

	if (salt[0] != '\0')
		return salt;
	strlcpy(salt, "xx", sizeof(salt));
	setpwent();
	while ((pw = getpwent()) != NULL) {
		if ((passwd = shadow_pw(pw)) == NULL)
			continue;
		if (passwd[0] == '$' && (p = strrchr(passwd+1, '$')) != NULL) {
			typelen = p - passwd + 1;
			strlcpy(salt, passwd, MINIMUM(typelen, sizeof(salt)));
			explicit_bzero(passwd, strlen(passwd));
			goto out;
		}
	}
 out:
	endpwent();
	return salt;
}

char *
xcrypt(const char *password, const char *salt)
{
	char *crypted;

	/*
	 * If we don't have a salt we are encrypting a fake password for
	 * for timing purposes.  Pick an appropriate salt.
	 */
	if (salt == NULL)
		salt = pick_salt();

#if defined(__hpux) && !defined(HAVE_SECUREWARE)
	if (iscomsec())
		crypted = bigcrypt(password, salt);
	else
		crypted = crypt(password, salt);
# elif defined(HAVE_SECUREWARE)
	crypted = bigcrypt(password, salt);
# else
	crypted = crypt(password, salt);
#endif

	return crypted;
}

/*
 * Handle shadowed password systems in a cleaner way for portable
 * version.
 */

char *
shadow_pw(struct passwd *pw)
{
	char *pw_password = pw->pw_passwd;

# if defined(HAVE_SHADOW_H) && !defined(DISABLE_SHADOW)
	struct spwd *spw = getspnam(pw->pw_name);

	if (spw != NULL)
		pw_password = spw->sp_pwdp;
# endif

#ifdef USE_LIBIAF
	return(get_iaf_password(pw));
#endif

# if defined(HAVE_GETPWANAM) && !defined(DISABLE_SHADOW)
	struct passwd_adjunct *spw;
	if (issecure() && (spw = getpwanam(pw->pw_name)) != NULL)
		pw_password = spw->pwa_passwd;
# elif defined(HAVE_SECUREWARE)
	struct pr_passwd *spw = getprpwnam(pw->pw_name);

	if (spw != NULL)
		pw_password = spw->ufld.fd_encrypt;
# endif

	return pw_password;
}
