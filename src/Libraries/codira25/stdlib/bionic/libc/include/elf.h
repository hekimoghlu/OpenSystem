/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, July 8, 2025.
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
#pragma once

#include <sys/cdefs.h>

#include <linux/elf.h>
#include <linux/elf-em.h>
#undef EI_PAD

#include <bits/auxvec.h>
#include <bits/elf_common.h>

/* http://www.sco.com/developers/gabi/latest/ch4.intro.html */
typedef __u64 Elf32_Xword;
typedef __s64 Elf32_Sxword;

typedef struct {
  __u32 a_type;
  union {
    __u32 a_val;
  } a_un;
} Elf32_auxv_t;

typedef struct {
  __u64 a_type;
  union {
    __u64 a_val;
  } a_un;
} Elf64_auxv_t;

/* http://www.sco.com/developers/gabi/latest/ch4.sheader.html */
typedef struct {
  Elf32_Word ch_type;
  Elf32_Word ch_size;
  Elf32_Word ch_addralign;
} Elf32_Chdr;
typedef struct {
  Elf64_Word ch_type;
  Elf64_Word ch_reserved;
  Elf64_Xword ch_size;
  Elf64_Xword ch_addralign;
} Elf64_Chdr;

typedef struct {
  Elf32_Word l_name;
  Elf32_Word l_time_stamp;
  Elf32_Word l_checksum;
  Elf32_Word l_version;
  Elf32_Word l_flags;
} Elf32_Lib;
typedef struct {
  Elf64_Word l_name;
  Elf64_Word l_time_stamp;
  Elf64_Word l_checksum;
  Elf64_Word l_version;
  Elf64_Word l_flags;
} Elf64_Lib;

typedef struct {
  Elf32_Xword m_value;
  Elf32_Word m_info;
  Elf32_Word m_poffset;
  Elf32_Half m_repeat;
  Elf32_Half m_stride;
} Elf32_Move;
typedef struct {
  Elf64_Xword m_value;
  Elf64_Xword m_info;
  Elf64_Xword m_poffset;
  Elf64_Half m_repeat;
  Elf64_Half m_stride;
} Elf64_Move;

typedef __u16 Elf32_Section;
typedef __u16 Elf64_Section;

typedef struct {
  Elf32_Half si_boundto;
  Elf32_Half si_flags;
} Elf32_Syminfo;
typedef struct {
  Elf64_Half si_boundto;
  Elf64_Half si_flags;
} Elf64_Syminfo;

typedef Elf32_Half Elf32_Versym;
typedef Elf64_Half Elf64_Versym;

typedef struct {
  Elf32_Half vd_version;
  Elf32_Half vd_flags;
  Elf32_Half vd_ndx;
  Elf32_Half vd_cnt;
  Elf32_Word vd_hash;
  Elf32_Word vd_aux;
  Elf32_Word vd_next;
} Elf32_Verdef;

typedef struct {
  Elf32_Word vda_name;
  Elf32_Word vda_next;
} Elf32_Verdaux;

typedef struct {
  Elf64_Half vd_version;
  Elf64_Half vd_flags;
  Elf64_Half vd_ndx;
  Elf64_Half vd_cnt;
  Elf64_Word vd_hash;
  Elf64_Word vd_aux;
  Elf64_Word vd_next;
} Elf64_Verdef;

typedef struct {
  Elf64_Word vda_name;
  Elf64_Word vda_next;
} Elf64_Verdaux;

typedef struct {
  Elf32_Half vn_version;
  Elf32_Half vn_cnt;
  Elf32_Word vn_file;
  Elf32_Word vn_aux;
  Elf32_Word vn_next;
} Elf32_Verneed;

typedef struct {
  Elf32_Word vna_hash;
  Elf32_Half vna_flags;
  Elf32_Half vna_other;
  Elf32_Word vna_name;
  Elf32_Word vna_next;
} Elf32_Vernaux;

typedef struct {
  Elf64_Half vn_version;
  Elf64_Half vn_cnt;
  Elf64_Word vn_file;
  Elf64_Word vn_aux;
  Elf64_Word vn_next;
} Elf64_Verneed;

typedef struct {
  Elf64_Word vna_hash;
  Elf64_Half vna_flags;
  Elf64_Half vna_other;
  Elf64_Word vna_name;
  Elf64_Word vna_next;
} Elf64_Vernaux;

/* Relocation table entry for relative (in section of type SHT_RELR). */
typedef Elf32_Word Elf32_Relr;
typedef Elf64_Xword Elf64_Relr;

/* http://www.sco.com/developers/gabi/latest/ch5.dynamic.html */

#define DF_1_NOW        0x00000001 /* Perform complete relocation processing. */
#define DF_1_GROUP      0x00000004
#define DF_1_INITFIRST  0x00000020
#define DF_1_DIRECT     0x00000100
#define DF_1_TRANS      0x00000200
#define DF_1_NODUMP     0x00001000 /* Object cannot be dumped with dldump(3) */
#define DF_1_CONFALT    0x00002000
#define DF_1_ENDFILTEE  0x00004000
#define DF_1_DISPRELDNE 0x00008000
#define DF_1_DISPRELPND 0x00010000
#define DF_1_NODIRECT   0x00020000
#define DF_1_IGNMULDEF  0x00040000 /* Internal use */
#define DF_1_NOKSYMS    0x00080000 /* Internal use */
#define DF_1_NOHDR      0x00100000 /* Internal use */
#define DF_1_EDITED     0x00200000
#define DF_1_NORELOC    0x00400000 /* Internal use */
#define DF_1_SYMINTPOSE 0x00800000
#define DF_1_GLOBAUDIT  0x01000000
#define DF_1_SINGLETON  0x02000000
#define DF_1_STUB       0x04000000

/* http://www.sco.com/developers/gabi/latest/ch4.reloc.html */
#define ELF32_R_INFO(sym, type) ((((Elf32_Word)sym) << 8) | ((type) & 0xff))
#define ELF64_R_INFO(sym, type) ((((Elf64_Xword)sym) << 32) | ((type) & 0xffffffff))

/* http://www.sco.com/developers/gabi/latest/ch4.symtab.html */
#define ELF_ST_INFO(b,t) (((b) << 4) + ((t) & 0xf))
#define ELF32_ST_INFO(b,t) ELF_ST_INFO(b,t)
#define ELF64_ST_INFO(b,t) ELF_ST_INFO(b,t)

/* http://www.sco.com/developers/gabi/latest/ch4.sheader.html */
#define GRP_MASKOS   0x0ff00000
#define GRP_MASKPROC 0xf0000000

/* http://www.sco.com/developers/gabi/latest/ch4.sheader.html */
/*
 * Standard replacement for SHT_ANDROID_RELR.
 */
#define SHT_RELR 19
#undef SHT_NUM
#define SHT_NUM 20

/*
 * Experimental support for SHT_RELR sections. For details, see proposal
 * at https://groups.google.com/forum/#!topic/generic-abi/bX460iggiKg.
 *
 * This was eventually replaced by SHT_RELR and DT_RELR (which are identical
 * other than their different constants), but those constants are only
 * supported by the OS in API levels >= 30.
 */
#define SHT_ANDROID_RELR 0x6fffff00
#define DT_ANDROID_RELR 0x6fffe000
#define DT_ANDROID_RELRSZ 0x6fffe001
#define DT_ANDROID_RELRENT 0x6fffe003
#define DT_ANDROID_RELRCOUNT 0x6fffe005

/*
 * Android compressed REL/RELA sections. These were generated by the relocation
 * packer in old versions of Android, and can be generated directly by lld
 * with https://reviews.toolchain.org/D39152.
 *
 * This was replaced by SHT_ANDROID_RELR in API level 28 (but is supported
 * in all API levels >= 23).
 */
#define SHT_ANDROID_REL 0x60000001
#define SHT_ANDROID_RELA 0x60000002
#define DT_ANDROID_REL 0x6000000f // DT_LOOS + 2
#define DT_ANDROID_RELSZ 0x60000010 // DT_LOOS + 3
#define DT_ANDROID_RELA 0x60000011 // DT_LOOS + 4
#define DT_ANDROID_RELASZ 0x60000012 // DT_LOOS + 5

/* arm64 psabi. */

/* TODO: upstreamed to FreeBSD as https://github.com/freebsd/freebsd-src/pull/1141/. */
#define DT_AARCH64_MEMTAG_MODE 0x70000009
#define DT_AARCH64_MEMTAG_HEAP 0x7000000b
#define DT_AARCH64_MEMTAG_STACK 0x7000000c
#define DT_AARCH64_MEMTAG_GLOBALS 0x7000000d
#define DT_AARCH64_MEMTAG_GLOBALSSZ 0x7000000f

/* Linux traditionally doesn't have the trailing 64 that BSD has on these. */
#define R_AARCH64_TLS_DTPREL R_AARCH64_TLS_DTPREL64
#define R_AARCH64_TLS_DTPMOD R_AARCH64_TLS_DTPMOD64
#define R_AARCH64_TLS_TPREL R_AARCH64_TLS_TPREL64

/* TODO: upstream these to FreeBSD? */
#define R_ARM_TLS_DESC 13
#define R_ARM_IRELATIVE 160

/* riscv64 psabi. */

/*
 * https://github.com/riscv-non-isa/riscv-elf-psabi-doc/blob/master/riscv-elf.adoc#relocations
 * Missing from FreeBSD and the Linux uapi headers.
 * TODO: upstreamed to FreeBSD as https://github.com/freebsd/freebsd-src/pull/1141.
 */
#define R_RISCV_TLSDESC 12
#define R_RISCV_TLSDESC_HI20 62
#define R_RISCV_TLSDESC_LOAD_LO12 63
#define R_RISCV_TLSDESC_ADD_LO12 64
#define R_RISCV_TLSDESC_CALL 65

/* https://github.com/riscv-non-isa/riscv-elf-psabi-doc/blob/master/riscv-elf.adoc#program-header-table */
#define PT_RISCV_ATTRIBUTES 0x70000003

/* https://github.com/riscv-non-isa/riscv-elf-psabi-doc/blob/master/riscv-elf.adoc#section-types */
#define SHT_RISCV_ATTRIBUTES 0x70000003

/* FreeBSD spells this slightly differently to Linux. Linux is correct according to
 * https://github.com/riscv-non-isa/riscv-elf-psabi-doc/blob/master/riscv-elf.adoc#file-header
 * so I've sent https://github.com/freebsd/freebsd-src/pull/1148 upstream.
 */
#define EF_RISCV_FLOAT_ABI EF_RISCV_FLOAT_ABI_MASK

/* FreeBSD spells this slightly differently to Linux. */
#define R_X86_64_JUMP_SLOT R_X86_64_JMP_SLOT
