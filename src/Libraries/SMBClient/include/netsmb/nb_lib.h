/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, February 17, 2024.
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
#ifndef _NETSMB_NB_LIB_H_
#define	_NETSMB_NB_LIB_H_


/*
 * resource record
 */
struct nbns_rr {
	u_char *	rr_name;	/* compressed NETBIOS name */
	uint16_t	rr_type;
	uint16_t	rr_class;
	uint32_t	rr_ttl;
	uint16_t	rr_rdlength;
	u_char *	rr_data;
};

/*
 * NetBIOS name return
 */
struct nbns_nr {
	char		nr_name[NB_NAMELEN];
	uint16_t	nr_beflags; /* Big endian, from network */
};

#define NBRQF_BROADCAST		0x0001

#define NBNS_GROUPFLG 0x8000


struct nb_ifdesc {
	int		id_flags;
	struct in_addr	id_addr;
	struct in_addr	id_mask;
	char		id_name[16];	/* actually IFNAMSIZ */
	struct nb_ifdesc * id_next;
};

struct sockaddr;

__BEGIN_DECLS

/* new flag UCflag. 1=uppercase,0=don't */
void nb_name_encode(struct nb_name *, u_char *);
int nb_encname_len(const char *);

int nb_sockaddr(struct sockaddr *peer, const char *name, unsigned type, 
				struct sockaddr **dst);
void convertToNetBIOSaddr(struct sockaddr_storage *storage, const char *name);

int resolvehost(const char *name, CFMutableArrayRef *outAddressArray, char *netbios_name, 
				uint16_t port,  int allowLocalConn, int tryBothPorts);
int findReachableAddress(CFMutableArrayRef addressArray, uint16_t *cancel, struct connectAddress **dest);
int nbns_resolvename(struct nb_ctx *ctx, struct smb_prefs *prefs, const char *name, 
					 uint8_t nodeType, CFMutableArrayRef *outAddressArray, uint16_t port, 
					 int allowLocalConn, int tryBothPorts, uint16_t *cancel);
int nbns_getnodestatus(struct sockaddr *targethost, struct nb_ctx *ctx,
					   struct smb_prefs *prefs, uint16_t *cancel, char *nbt_server, 
					   char *workgroup, CFMutableArrayRef nbrrArray);
int isLocalIPAddress(struct sockaddr *, uint16_t port, int allowLocalConn);
int isIPv6NumericName(const char *name);
int nb_enum_if(struct nb_ifdesc **, int);
int nb_error_to_errno(int error);

int nb_ctx_resolve(struct nb_ctx *ctx, CFArrayRef WINSAddresses);
__END_DECLS

#endif /* !_NETSMB_NB_LIB_H_ */
