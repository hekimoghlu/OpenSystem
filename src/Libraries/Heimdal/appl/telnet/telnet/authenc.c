/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, August 30, 2025.
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
#include "telnet_locl.h"

RCSID("$Id$");

#if	defined(AUTHENTICATION) || defined(ENCRYPTION)
int
telnet_net_write(unsigned char *str, int len)
{
	if (NETROOM() > len) {
		ring_supply_data(&netoring, str, len);
		if (str[0] == IAC && str[1] == SE)
			printsub('>', &str[2], len-2);
		return(len);
	}
	return(0);
}

void
net_encrypt(void)
{
#if	defined(ENCRYPTION)
	if (encrypt_output)
		ring_encrypt(&netoring, encrypt_output);
	else
		ring_clearto(&netoring);
#endif
}

int
telnet_spin(void)
{
    int ret = 0;

    scheduler_lockout_tty = 1;
    if (Scheduler(0) == -1)
	ret = 1;
    scheduler_lockout_tty = 0;

    return ret;

}

char *
telnet_getenv(const char *val)
{
	return((char *)env_getvalue((unsigned char *)val));
}

char *
telnet_gets(char *prompt, char *result, int length, int echo)
{
	int om = globalmode;
	char *res;

	TerminalNewMode(-1);
	if (echo) {
		printf("%s", prompt);
		res = fgets(result, length, stdin);
	} else if ((res = getpass(prompt))) {
		strlcpy(result, res, length);
		res = result;
	}
	TerminalNewMode(om);
	return(res);
}
#endif
