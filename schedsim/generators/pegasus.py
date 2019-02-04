"""
Generators for synthetic workflows from the Pegasus workflow gallery
(https://pegasus.isi.edu/workflow_gallery/index.php).
"""
import collections
import random

from .utils import normal, gen_level, join_level
from ..common import TaskGraph


def montage(width=50):
    tg = TaskGraph()

    gen_width = int(width / 4)

    top_levels = [tg.new_task(name="mProjectPP-{}".format(i),
                              duration=normal(15, 3),
                              expected_duration=15,
                              outputs=[normal(4, 1), normal(4, 1)])
                  for i in range(gen_width)]

    middle = [tg.new_task(name="mDiffFit-{}".format(i),
                          duration=normal(10, 1),
                          expected_duration=10,
                          outputs=[normal(0.2, 0.1), normal(0.2, 0.1)])
              for i in range(width)]

    # connect middle to top
    for (i, task) in enumerate(top_levels):
        i *= 4
        start = max(0, i - 2)
        end = min(len(middle), i + 2)

        for m in middle[start:end]:
            m.add_input(random.choice(task.outputs))

    concat = tg.new_task(name="mConcatFit",
                         duration=normal(13, 2),
                         expected_duration=13,
                         output_size=0.01)
    for t in middle:
        for o in t.outputs:
            concat.add_input(o)

    model = tg.new_task(name="mBgModel",
                        duration=normal(11, 1),
                        expected_duration=11,
                        output_size=0.01)
    model.add_input(concat)

    backgrounds = [tg.new_task(name="mBackground-{}".format(i),
                               duration=normal(10, 1),
                               expected_duration=10,
                               outputs=[normal(4, 1), normal(4, 1)])
                   for i in range(gen_width)]

    for (i, bg) in enumerate(backgrounds):
        bg.add_input(model)
        bg.add_input(random.choice(top_levels[i].outputs))

    img = tg.new_task("mImgTbl", duration=8, expected_duration=8)
    for t in backgrounds:
        for o in t.outputs:
            img.add_input(o)

    return tg


def cybershake(width=50):
    tg = TaskGraph()

    top = [tg.new_task(name="ExtractGST-{}".format(i),
                       duration=normal(100, 50),
                       expected_duration=100,
                       outputs=[normal(220, 80), normal(220, 80)])
           for i in range(2)]
    synthesis = gen_level(tg, width, "SeismogramSynthesis-{}", lambda: normal(45, 10),
                          lambda: normal(0.02, 0.01), 45)
    half = int(width / 2)
    for (i, task) in enumerate(synthesis):
        synthesis[i].add_inputs(top[int(i / half)].outputs)

    zip_seis = tg.new_task(name="ZipSeis",
                           duration=normal(0.5, 0.2),
                           expected_duration=0.5,
                           output_size=normal(0.2, 0.1))
    zip_seis.add_inputs(synthesis)

    peakval = gen_level(tg, width, "PeakValCalcOkaya-{}", lambda: normal(1, 0.5), lambda: 0.01, 1)
    join_level(synthesis, peakval)

    zip_psa = tg.new_task(name="ZipPSA",
                          duration=normal(0.5, 0.2),
                          expected_duration=0.5,
                          output_size=0.1)
    zip_psa.add_inputs(peakval)

    return tg


def epigenomics(width=50):
    tg = TaskGraph()

    split = tg.new_task(name="fastqSplit",
                        duration=normal(35, 5),
                        expected_duration=35,
                        outputs=[normal(10, 5) for _ in range(width)])

    filter = gen_level(tg, width, "filter_Contams-{}",
                       lambda: normal(1.5, 1),
                       lambda: [normal(5, 2) for _ in range(2)], 1.5)
    for (i, t) in enumerate(filter):
        t.add_input(split.outputs[i])

    sol2 = gen_level(tg, width, "sol2sanger-{}", lambda: normal(0.3, 0.2),
                     lambda: normal(5, 2), 0.3)
    join_level(filter, sol2)

    fastg = gen_level(tg, width, "fastq2bfq-{}", lambda: normal(1, 0.2),
                      lambda: normal(1, 0.5), 1)
    join_level(sol2, fastg)

    map_chr = gen_level(tg, width, "map_chr-{}", lambda: normal(15000, 2500),
                        lambda: normal(1, 0.5), 15000)
    join_level(fastg, map_chr)

    merge = tg.new_task(name="mapMerge", duration=normal(10, 2),
                        expected_duration=10,
                        outputs=[normal(20, 2), 0.5, 0.1])
    merge.add_inputs(map_chr)

    maqindex = tg.new_task(name="maqindex", duration=0.02,
                           expected_duration=0.02,
                           output_size=0.4)
    maqindex.add_input(merge.outputs[0])
    tg.new_task(name="pileup",
                duration=normal(6000, 200),
                expected_duration=6000,
                output_size=3).add_input(maqindex)

    return tg


def ligo(widths=(20, 10)):
    if not isinstance(widths, collections.Iterable):
        widths = (widths, )

    tg = TaskGraph()

    def create_subgraph(w):
        parent = None

        for _ in range(2):
            top = gen_level(tg, w, "TmpItBank-{}", lambda: normal(19, 0.1),
                            lambda: normal(0.9, 0.1), 19)
            if parent:
                for t in top:
                    t.add_input(parent)

            inspiral = gen_level(tg, w, "Inspiral-{}", lambda: normal(400, 100),
                                 lambda: normal(0.4, 0.1), 400)
            join_level(top, inspiral)
            thinca = tg.new_task("Thinca", duration=normal(5, 1), output_size=0.02)
            thinca.add_inputs(inspiral)
            parent = thinca

    [create_subgraph(w) for w in widths]

    return tg


def sipht(copies=2):
    tg = TaskGraph()

    def create_subgraph():
        parsers = gen_level(tg, 20, "Parser-{}",
                            lambda: normal(1, 0.2),
                            lambda: normal(0.1, 0.02),
                            1)
        parser_concat = tg.new_task(name="Parser-concat",
                                    duration=0.05,
                                    expected_duration=0.05,
                                    output_size=normal(1.5, 0.1))
        parser_concat.add_inputs(parsers)

        find_term = tg.new_task(name="Findterm",
                                duration=normal(1800, 100),
                                expected_duration=1800,
                                outputs=[0, 0, normal(20, 1), normal(0.2, 0.1), 0,
                                         normal(4, 1), normal(4, 1), normal(6, 1)])
        rna_motif = tg.new_task(name="RNAMotif",
                                duration=normal(38, 2),
                                expected_duration=38,
                                outputs=[0, 1])
        transterm = tg.new_task(name="Transterm",
                                duration=normal(50, 2),
                                expected_duration=50,
                                outputs=[0, 0.5])
        blast = tg.new_task(name="Blast",
                            duration=normal(1900, 100),
                            expected_duration=1900,
                            outputs=[0, 0, 4.5])

        top = [find_term, rna_motif, transterm, blast]

        srna = tg.new_task(name="SRNA",
                           duration=normal(450, 50),
                           expected_duration=450,
                           outputs=[normal(0.5, 0.2) for _ in range(10)])
        for t in top:
            srna.add_inputs(t.outputs)
        output = srna.outputs[-1]

        blast = tg.new_task(name="BlastQRNA",
                            duration=normal(1200, 100),
                            expected_duration=1200,
                            outputs=[normal(0.5, 1) for _ in range(8)])
        blast2 = tg.new_task(name="Blast_paralogues",
                             duration=normal(5, 1),
                             expected_duration=5,
                             output_size=0.9)
        ffn_parse = tg.new_task(name="FFN_Parse",
                                duration=normal(1, 1),
                                expected_duration=1,
                                output_size=0.9)
        blast_synteny = tg.new_task(name="Blast_synteny",
                                    duration=normal(30, 5),
                                    expected_duration=30,
                                    output_size=1)
        blast_synteny.add_input(ffn_parse)
        blast_candidate = tg.new_task(name="Blast_candidate",
                                      duration=normal(6, 1),
                                      expected_duration=6,
                                      outputs=[0] * 3)
        level = [blast, blast2, ffn_parse, blast_synteny, blast_candidate]
        for t in level:
            t.add_input(output)

        annotate = tg.new_task(name="SRNA_annotate",
                               duration=normal(1, 2),
                               expected_duration=1,
                               outputs=[normal(0.5, 0.2) for _ in range(8)])
        annotate.add_input(output)
        for t in level:
            annotate.add_input(t.outputs[0])
        annotate.add_input(parser_concat)

    [create_subgraph() for _ in range(copies)]

    return tg
