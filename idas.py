# information to be collected

# import : function call & library name (.idata section)
# string : string values from .rdata section (.rdata section)
# basic blocks
# data section

import idaapi
import json
import idautils
import idc

idaapi.autoWait()

file_prefix = os.path.splitext(idc.GetIdbPath())[0]
meta_file = file_prefix + ".json"
output_file = file_prefix + ".tagged.json"
info = idaapi.get_inf_structure()
sha256_calculated = idaapi.retrieve_input_file_sha256().lower()
sha256_stated = os.path.split(file_prefix)[-1].lower()

imports_funcs = set()
imports_libs = set()

def imp_cb(ea, name, ord):
    if name:
        imports_funcs.add(str(name).strip().lower())
    return True


def proc_i64():
    # commented out for serving.
    # if not os.path.exists(meta_file):
    #     print('meta file non-exist')
    #     return
    if info.procName.lower() != "metapc":
        print('arch {} is not metapc'.format(info.procName))
        return
    if sha256_stated.strip() != sha256_calculated.strip():
        print('sha256 not matched stated {} vs calculated {}'.format(sha256_stated, sha256_calculated))
        return
     
    segs = {
        'sha256':sha256_calculated,
        'seg':[],
        'rsk':{}, 
        'imp_m':[],
        'imp_f':[],
        'str':[],
        'asm':[],
        'data': {},
        'byte': [] # bytes of the input file.
    }
    
    # read the original input file:
    bin_file = file_prefix
    if not os.path.exists(bin_file):
        bin_file = bin_file + ".bin"
    segs['byte'] = [ord(b) for b in open(bin_file, "rb").read()]
    
    # get strings
    segs['str'] = [str(st).decode(encoding='UTF-8',errors='ignore').strip().lower() for st in idautils.Strings() if len(str(st).strip()) > 0]
    
    
    # processing imports:
    nimps = idaapi.get_import_module_qty()
    for i in xrange(0, nimps):
        name = idaapi.get_import_module_name(i)
        if not name:
            print "Failed to get import module name for #%d" % i
            continue
        
        imports_libs.add(name.strip().lower())
        idaapi.enum_import_names(i, imp_cb)
    
    segs['imp_m'] = list(imports_libs)
    segs['imp_f'] = list(imports_funcs)
    
    # processing code segments:
    seg_code = segs['asm']
    for seg_ea in Segments() :
        seg = idaapi.getseg(seg_ea)
        cls = idaapi.get_segm_class(seg)
        nme = idaapi.get_segm_name(seg)
        segs['seg'].append((nme,cls))
        sea = seg.startEA
        eea = seg.endEA
        if cls == 'CODE' or 'text' in nme:
            blk = {'name':None, 'ins':[]}
            for head in Heads(sea, eea):
                name = idc.Name(head).strip()
                if len(name) > 0:
                    if blk['name'] is not None and len(blk['ins']) > 0:
                        seg_code.append(blk)
                    blk = {'name': name, 'ins':[]}
                tline = list()
                tline.append(
                        str(hex(head)).rstrip("L").upper().replace("0X", "0x"))
                mnem = idc.GetMnem(head)
                if mnem is None or len(mnem.strip()) < 1:
                    continue
                mnem = idc.GetDisasm(head).split()[0]
                tline.append(mnem)
                for i in range(5):
                    opd = idc.GetOpnd(head, i)
                    if opd == "":
                        continue
                    tline.append(opd)
                blk['ins'].append(tline)
            if blk['name'] is not None and len(blk['ins']) > 0:
                seg_code.append(blk)
        elif cls == 'DATA' or cls == 'CONST':
            byte_array = []
            start = idc.GetSegmentAttr(sea, idc.SEGATTR_START)
            end = idc.GetSegmentAttr(sea, idc.SEGATTR_END)
            while start < end:
                b = idc.Byte(start)
                byte_array.append(b)
                start += 1
            if len(byte_array) > 0:
                segs['data'][nme] =  byte_array
            
                
    segs['imp_m'].sort()
    segs['imp_f'].sort()
    segs['str'].sort()
    # load meta data:
    if os.path.exists(meta_file):
        with open(meta_file) as f:
            rp = json.load(f)
            segs['rsk'] = rp['signatures']
    
    with open(output_file, 'w') as f:
        json.dump(segs, f) #indent=4, separators=(',', ': ')
    
    return


proc_i64()
idc.Exit(0)
