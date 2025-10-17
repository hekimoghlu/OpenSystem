/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Tuesday, December 3, 2024.
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
#ifndef ELF_MACHINE_H_INCLUDED
#define ELF_MACHINE_H_INCLUDED

#define YASM_WRITE_32I_L(p, i) do {\
    assert(yasm_intnum_check_size(i, 32, 0, 2)); \
    yasm_intnum_get_sized(i, p, 4, 32, 0, 0, 0); \
    p += 4; } while (0)

#define YASM_WRITE_64I_L(p, i) do {\
    assert(yasm_intnum_check_size(i, 64, 0, 2)); \
    yasm_intnum_get_sized(i, p, 8, 64, 0, 0, 0); \
    p += 8; } while (0)

#define YASM_WRITE_64C_L(p, hi, lo) do {\
    YASM_WRITE_32_L(p, lo); \
    YASM_WRITE_32_L(p, hi); } while (0)

#define YASM_WRITE_64Z_L(p, i)          YASM_WRITE_64C_L(p, 0, i)

typedef int(*func_accepts_reloc)(size_t val, yasm_symrec *wrt);
typedef void(*func_write_symtab_entry)(unsigned char *bufp,
                                       elf_symtab_entry *entry,
                                       yasm_intnum *value_intn,
                                       yasm_intnum *size_intn);
typedef void(*func_write_secthead)(unsigned char *bufp, elf_secthead *shead);
typedef void(*func_write_secthead_rel)(unsigned char *bufp,
                                       elf_secthead *shead,
                                       elf_section_index symtab_idx,
                                       elf_section_index sindex);

typedef void(*func_handle_reloc_addend)(yasm_intnum *intn,
                                        elf_reloc_entry *reloc,
                                        unsigned long offset);
typedef unsigned int(*func_map_reloc_info_to_type)(elf_reloc_entry *reloc);
typedef void(*func_write_reloc)(unsigned char *bufp,
                                elf_reloc_entry *reloc,
                                unsigned int r_type,
                                unsigned int r_sym);
typedef void (*func_write_proghead)(unsigned char **bufpp,
                                    elf_offset secthead_addr,
                                    unsigned long secthead_count,
                                    elf_section_index shstrtab_index);

enum {
    ELF_SSYM_SYM_RELATIVE = 1 << 0,
    ELF_SSYM_CURPOS_ADJUST = 1 << 1,
    ELF_SSYM_THREAD_LOCAL = 1 << 2
};

typedef struct {
    const char *name;       /* should be something like ..name */
    const int sym_rel;      /* symbol or section-relative? */
    const unsigned int reloc;   /* relocation type */
    const unsigned int size;    /* legal data size */
} elf_machine_ssym;

struct elf_machine_handler {
    const char *arch;
    const char *machine;
    const char *reloc_section_prefix;
    const unsigned long symtab_entry_size;
    const unsigned long symtab_entry_align;
    const unsigned long reloc_entry_size;
    const unsigned long secthead_size;
    const unsigned long proghead_size;
    func_accepts_reloc accepts_reloc;
    func_write_symtab_entry write_symtab_entry;
    func_write_secthead write_secthead;
    func_write_secthead_rel write_secthead_rel;
    func_handle_reloc_addend handle_reloc_addend;
    func_map_reloc_info_to_type map_reloc_info_to_type;
    func_write_reloc write_reloc;
    func_write_proghead write_proghead;

    elf_machine_ssym *ssyms;            /* array of "special" syms */
    const size_t num_ssyms;             /* size of array */

    const int bits;                     /* usually 32 or 64 */
};

#endif /* ELF_MACHINE_H_INCLUDED */
