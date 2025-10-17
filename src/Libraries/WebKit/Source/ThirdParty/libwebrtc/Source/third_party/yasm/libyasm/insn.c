/*
 *
 * Copyright (c) NeXTHub Corporation. All Rights Reserved. 
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * Author: Tunjay Akbarli
 * Date: Saturday, October 22, 2022.
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
#include "util.h"

#include "libyasm-stdint.h"
#include "coretype.h"

#include "errwarn.h"
#include "expr.h"
#include "value.h"

#include "bytecode.h"
#include "insn.h"
#include "arch.h"


void
yasm_ea_set_segreg(yasm_effaddr *ea, uintptr_t segreg)
{
    if (!ea)
        return;

    if (segreg != 0 && ea->segreg != 0)
        yasm_warn_set(YASM_WARN_GENERAL,
                      N_("multiple segment overrides, using leftmost"));

    ea->segreg = segreg;
}

yasm_insn_operand *
yasm_operand_create_reg(uintptr_t reg)
{
    yasm_insn_operand *retval = yasm_xmalloc(sizeof(yasm_insn_operand));

    retval->type = YASM_INSN__OPERAND_REG;
    retval->data.reg = reg;
    retval->seg = 0;
    retval->targetmod = 0;
    retval->size = 0;
    retval->deref = 0;
    retval->strict = 0;

    return retval;
}

yasm_insn_operand *
yasm_operand_create_segreg(uintptr_t segreg)
{
    yasm_insn_operand *retval = yasm_xmalloc(sizeof(yasm_insn_operand));

    retval->type = YASM_INSN__OPERAND_SEGREG;
    retval->data.reg = segreg;
    retval->seg = 0;
    retval->targetmod = 0;
    retval->size = 0;
    retval->deref = 0;
    retval->strict = 0;

    return retval;
}

yasm_insn_operand *
yasm_operand_create_mem(/*@only@*/ yasm_effaddr *ea)
{
    yasm_insn_operand *retval = yasm_xmalloc(sizeof(yasm_insn_operand));

    retval->type = YASM_INSN__OPERAND_MEMORY;
    retval->data.ea = ea;
    retval->seg = 0;
    retval->targetmod = 0;
    retval->size = 0;
    retval->deref = 0;
    retval->strict = 0;
    retval->size = ea->data_len * 8;

    return retval;
}

yasm_insn_operand *
yasm_operand_create_imm(/*@only@*/ yasm_expr *val)
{
    yasm_insn_operand *retval;
    const uintptr_t *reg;

    reg = yasm_expr_get_reg(&val, 0);
    if (reg) {
        retval = yasm_operand_create_reg(*reg);
        yasm_expr_destroy(val);
    } else {
        retval = yasm_xmalloc(sizeof(yasm_insn_operand));
        retval->type = YASM_INSN__OPERAND_IMM;
        retval->data.val = val;
        retval->seg = 0;
        retval->targetmod = 0;
        retval->size = 0;
        retval->deref = 0;
        retval->strict = 0;
    }

    return retval;
}

yasm_insn_operand *
yasm_insn_ops_append(yasm_insn *insn, yasm_insn_operand *op)
{
    if (op) {
        insn->num_operands++;
        STAILQ_INSERT_TAIL(&insn->operands, op, link);
        return op;
    }
    return (yasm_insn_operand *)NULL;
}

void
yasm_insn_add_prefix(yasm_insn *insn, uintptr_t prefix)
{
    insn->prefixes =
        yasm_xrealloc(insn->prefixes,
                      (insn->num_prefixes+1)*sizeof(uintptr_t));
    insn->prefixes[insn->num_prefixes] = prefix;
    insn->num_prefixes++;
}

void
yasm_insn_add_seg_prefix(yasm_insn *insn, uintptr_t segreg)
{
    insn->segregs =
        yasm_xrealloc(insn->segregs, (insn->num_segregs+1)*sizeof(uintptr_t));
    insn->segregs[insn->num_segregs] = segreg;
    insn->num_segregs++;
}

void
yasm_insn_initialize(yasm_insn *insn)
{
    STAILQ_INIT(&insn->operands);

    insn->prefixes = NULL;
    insn->segregs = NULL;

    insn->num_operands = 0;
    insn->num_prefixes = 0;
    insn->num_segregs = 0;
}

void
yasm_insn_delete(yasm_insn *insn,
                 void (*ea_destroy) (/*@only@*/ yasm_effaddr *))
{
    if (insn->num_operands > 0) {
        yasm_insn_operand *cur, *next;

        cur = STAILQ_FIRST(&insn->operands);
        while (cur) {
            next = STAILQ_NEXT(cur, link);
            switch (cur->type) {
                case YASM_INSN__OPERAND_MEMORY:
                    ea_destroy(cur->data.ea);
                    break;
                case YASM_INSN__OPERAND_IMM:
                    yasm_expr_destroy(cur->data.val);
                    break;
                default:
                    break;
            }
            yasm_xfree(cur);
            cur = next;
        }
    }
    if (insn->num_prefixes > 0)
        yasm_xfree(insn->prefixes);
    if (insn->num_segregs > 0)
        yasm_xfree(insn->segregs);
}

void
yasm_insn_print(const yasm_insn *insn, FILE *f, int indent_level)
{
    const yasm_insn_operand *op;

    STAILQ_FOREACH (op, &insn->operands, link) {
        switch (op->type) {
            case YASM_INSN__OPERAND_REG:
                fprintf(f, "%*sReg=", indent_level, "");
                /*yasm_arch_reg_print(arch, op->data.reg, f);*/
                fprintf(f, "\n");
                break;
            case YASM_INSN__OPERAND_SEGREG:
                fprintf(f, "%*sSegReg=", indent_level, "");
                /*yasm_arch_segreg_print(arch, op->data.reg, f);*/
                fprintf(f, "\n");
                break;
            case YASM_INSN__OPERAND_MEMORY:
                fprintf(f, "%*sMemory=\n", indent_level, "");
                /*yasm_arch_ea_print(arch, op->data.ea, f, indent_level);*/
                break;
            case YASM_INSN__OPERAND_IMM:
                fprintf(f, "%*sImm=", indent_level, "");
                yasm_expr_print(op->data.val, f);
                fprintf(f, "\n");
                break;
        }
        fprintf(f, "%*sTargetMod=%lx\n", indent_level+1, "",
                (unsigned long)op->targetmod);
        fprintf(f, "%*sSize=%u\n", indent_level+1, "", op->size);
        fprintf(f, "%*sDeref=%d, Strict=%d\n", indent_level+1, "",
                (int)op->deref, (int)op->strict);
    }
}

void
yasm_insn_finalize(yasm_insn *insn)
{
    unsigned int i;
    yasm_insn_operand *op;
    yasm_error_class eclass;
    char *str, *xrefstr;
    unsigned long xrefline;

    /* Simplify the operands' expressions first. */
    for (i = 0, op = yasm_insn_ops_first(insn);
         op && i<insn->num_operands; op = yasm_insn_op_next(op), i++) {
        /* Check operand type */
        switch (op->type) {
            case YASM_INSN__OPERAND_MEMORY:
                /* Don't get over-ambitious here; some archs' memory expr
                 * parser are sensitive to the presence of *1, etc, so don't
                 * simplify reg*1 identities.
                 */
                if (op->data.ea)
                    op->data.ea->disp.abs =
                        yasm_expr__level_tree(op->data.ea->disp.abs, 1, 1, 0,
                                              0, NULL, NULL);
                if (yasm_error_occurred()) {
                    /* Add a pointer to where it was used to the error */
                    yasm_error_fetch(&eclass, &str, &xrefline, &xrefstr);
                    if (xrefstr) {
                        yasm_error_set_xref(xrefline, "%s", xrefstr);
                        yasm_xfree(xrefstr);
                    }
                    if (str) {
                        yasm_error_set(eclass, "%s in memory expression", str);
                        yasm_xfree(str);
                    }
                    return;
                }
                break;
            case YASM_INSN__OPERAND_IMM:
                op->data.val =
                    yasm_expr__level_tree(op->data.val, 1, 1, 1, 0, NULL,
                                          NULL);
                if (yasm_error_occurred()) {
                    /* Add a pointer to where it was used to the error */
                    yasm_error_fetch(&eclass, &str, &xrefline, &xrefstr);
                    if (xrefstr) {
                        yasm_error_set_xref(xrefline, "%s", xrefstr);
                        yasm_xfree(xrefstr);
                    }
                    if (str) {
                        yasm_error_set(eclass, "%s in immediate expression",
                                       str);
                        yasm_xfree(str);
                    }
                    return;
                }
                break;
            default:
                break;
        }
    }
}
