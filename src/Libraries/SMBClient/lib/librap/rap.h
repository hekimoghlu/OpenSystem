/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Thursday, July 24, 2025.
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
#ifndef _RAP_H_
#define _RAP_H_

/*
 * RAP error codes
 */
#define	SMB_ERROR_ACCESS_DENIED			5
#define	SMB_ERROR_NETWORK_ACCESS_DENIED	65
#define SMB_ERROR_MORE_DATA				234

#define get16(buf,ofs)		(*((uint16_t*)(&((uint8_t*)(buf))[ofs])))
#define get32(buf,ofs)		(*((uint32_t*)(&((uint8_t*)(buf))[ofs])))

#define getwle(buf,ofs)	OSSwapHostToLittleInt16(get16(buf,ofs)
#define getwbe(buf,ofs)	OSSwapHostToBigInt16(get16(buf,ofs)
#define getdle(buf,ofs)	OSSwapHostToLittleInt32(get32(buf,ofs)
#define getdbe(buf,ofs) OSSwapHostToBigInt32(get32(buf,ofs)

#define setwle(buf,ofs,val)	get16(buf,ofs)=OSSwapHostToLittleInt16(val)
#define setwbe(buf,ofs,val) get16(buf,ofs)=OSSwapHostToBigInt16(val)
#define setdle(buf,ofs,val) get32(buf,ofs)=OSSwapHostToLittleInt32(val)
#define setdbe(buf,ofs,val) get32(buf,ofs)=OSSwapHostToBigInt32(val)

struct smb_rap {
	char *		r_sparam;
	char *		r_nparam;
	char *		r_sdata;
	char *		r_ndata;
	char *		r_pbuf;		/* rq parameters */
	int		r_plen;		/* rq param len */
	char *		r_npbuf;
	char *		r_dbuf;		/* rq data */
	int		r_dlen;		/* rq data len */
	char *		r_ndbuf;
	uint32_t	r_result;
	char *		r_rcvbuf;
	int		r_rcvbuflen;
	int		r_entries;
};

struct smb_share_info_1 {
	char		shi1_netname[13];
	char		shi1_pad;
	uint16_t	shi1_type;
	uint32_t	shi1_remark;		/* char * */
};

__BEGIN_DECLS

int
RapNetShareEnum(SMBHANDLE inConnection, int sLevel, void **rBuffer, uint32_t *rBufferSize, 
				uint32_t *entriesRead, uint32_t *totalEntriesRead);
void RapNetApiBufferFree(void * bufptr);

__END_DECLS

#endif /* _RAP_H_ */
