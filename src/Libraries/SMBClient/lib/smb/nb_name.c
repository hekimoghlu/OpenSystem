/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 10, 2024.
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
#include <netsmb/netbios.h>
#include <sys/smb_byte_order.h>
#include <netsmb/smb_lib.h>
#include <netsmb/nb_lib.h>

static int nb_snballoc(int namelen, struct sockaddr_nb **dst)
{
	struct sockaddr_nb *snb;
	int slen;

	slen = namelen + (int)sizeof(*snb) - (int)sizeof(snb->snb_name);
	snb = malloc(slen);
	if (snb == NULL)
		return ENOMEM;
	bzero(snb, slen);
	snb->snb_family = AF_NETBIOS;
	snb->snb_len = slen;
	*dst = snb;
	return 0;
}

/*
 * Create a NetBIOS address, we expect the name to already be in the correct
 * case.
 */
int nb_sockaddr(struct sockaddr *peer, const char *name, unsigned type,  
				struct sockaddr **dst)

{
	struct nb_name nn;
	struct nb_name *np = &nn;
	struct sockaddr_nb *snb;
	int nmlen, error;

	if (peer && (peer->sa_family != AF_INET))
		return EPROTONOSUPPORT;
	
	strlcpy((char *)nn.nn_name, name, sizeof(nn.nn_name));
	nn.nn_type = type;
	
	nmlen = NB_ENCNAMELEN + 2;
	error = nb_snballoc(nmlen, &snb);
	if (error)
		return error;
	nb_name_encode(np, snb->snb_name);
	if (peer)
		memcpy(&snb->snb_addrin, peer, peer->sa_len);
	*dst = (struct sockaddr *)snb;
	return 0;
}

/* 
 * Convert a sockaddr into a sockaddr_nb. We always assume the type is server. On 
 * error we leave the sockaddr in its original state.
 */
void convertToNetBIOSaddr(struct sockaddr_storage *storage, const char *name)
{
	struct sockaddr *peer = (struct sockaddr *)storage;
	struct sockaddr *dst = NULL;
	
	if (nb_sockaddr(peer, name, NBT_SERVER, &dst) == 0) {
        if ((peer != NULL) && (dst != NULL)) {
            memcpy(peer, dst, dst->sa_len);
        }
        
        if (dst != NULL) {
            free(dst);
        }
	}
}

int
nb_encname_len(const char *str)
{
	const u_char *cp = (const u_char *)str;
	int len, blen;

	if ((cp[0] & 0xc0) == 0xc0)
		return -1;	/* first two bytes are offset to name */

	len = 1;
	for (;;) {
		blen = *cp;
		if (blen++ == 0)
			break;
		len += blen;
		cp += blen;
	}
	return len;
}

/* B4BP (7/23/01 sent to BP) endian fix! */
#define	NBENCODE(c)	(htoles((u_short)(((u_char)(c) >> 4) | \
			 (((u_char)(c) & 0xf) << 8)) + 0x4141))

static void
memsetw(char *dst, int n, u_short word)
{
	while (n--) {
		*(u_short*)((void *)dst) = word;
		dst += 2;
	}
}

/*
 * We never uppercase in this routine any more. If the calling process wants
 * it uppercased then it shoud make sure nn_name is uppercase before entering
 * this routine. There is no way for us to get that correct in the routine.
 */
void nb_name_encode(struct nb_name *np, u_char *dst)
{
	u_char *name;
	u_char *cp = dst;
	int i;

	*cp++ = NB_ENCNAMELEN;
	name = np->nn_name;
	if (name[0] == '*' && name[1] == 0) {
		*(u_short*)((void *)cp) = NBENCODE('*');
		memsetw((char *)cp + 2, NB_NAMELEN - 1, NBENCODE((char)0));
		cp += NB_ENCNAMELEN;
	} else {
		/* freebsd bug: system names must be truncated to 15 chars not 16 */
		for (i = 0; *name && i < NB_NAMELEN - 1; i++, cp += 2, name++)
				*(u_short*)((void *)cp) = NBENCODE(*name);

		i = NB_NAMELEN - i - 1;
		if (i > 0) {
			memsetw((char *)cp, i, NBENCODE(' '));
			cp += i * 2;
		}
		*(u_short*)((void *)cp) = NBENCODE(np->nn_type);
		cp += 2;
	}
	*cp = 0;
}

