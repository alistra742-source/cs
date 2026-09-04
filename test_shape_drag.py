import io, math, random, unittest
from PIL import Image, ImageDraw, ImageFilter
import shape_drag

def scene(target_petals=5, decoy=4, seed=0, piece_at=(285,85)):
    rnd=random.Random(seed)
    bg=Image.new("RGB",(560,360)); dd=ImageDraw.Draw(bg)
    for i in range(0,560,8):
        dd.rectangle((i,0,i+8,360),fill=(int(60+120*abs(math.sin(i/90+seed))),
                                         int(110+90*abs(math.sin(i/50+1))),
                                         int(90+80*abs(math.cos(i/70)))))
    bg=bg.filter(ImageFilter.GaussianBlur(18)); d=ImageDraw.Draw(bg)
    def g(cx,cy,r,p,rot,c=(60,90,140),w=3):
        pts=[]
        for i in range(p*2):
            a=rot+i*math.pi/p; rr=r if i%2==0 else r*0.42
            pts.append((cx+rr*math.cos(a),cy+rr*math.sin(a)))
        d.polygon(pts,outline=c,width=w)
    spots=[(70,140),(480,150),(200,260),(400,255),(300,195)]
    ti=rnd.randrange(len(spots))
    for i,(x,y) in enumerate(spots):
        g(x,y,32,target_petals if i==ti else decoy,rnd.uniform(0,2))
    d.rectangle((piece_at[0]-45,piece_at[1]-45,piece_at[0]+45,piece_at[1]+45),fill=(235,238,230))
    g(piece_at[0],piece_at[1],30,target_petals,rnd.uniform(0,2),c=(200,190,40),w=4)
    b=io.BytesIO(); bg.save(b,format="PNG")
    return b.getvalue(), spots[ti]

class T(unittest.TestCase):
    def test_matches_target_across_layouts(self):
        hits=0
        for s in range(10):
            img,tgt=scene(5 if s%2 else 6, 4, s)
            r=shape_drag.solve_shape_drag(img)
            self.assertIsNotNone(r, f"seed {s}")
            tx,ty=r["to"][0]*560, r["to"][1]*360
            if abs(tx-tgt[0])<45 and abs(ty-tgt[1])<45: hits+=1
        self.assertGreaterEqual(hits,9,f"only {hits}/10")

    def test_source_is_the_loose_piece(self):
        img,_=scene(5,4,3)
        r=shape_drag.solve_shape_drag(img)
        fx,fy=r["from"][0]*560, r["from"][1]*360
        self.assertLess(abs(fx-285),40)
        self.assertLess(abs(fy-85),40)

    def test_from_and_to_differ(self):
        img,_=scene(5,4,1)
        r=shape_drag.solve_shape_drag(img)
        self.assertNotEqual(r["from"], r["to"])

    def test_rotation_invariant_signature(self):
        import numpy as np
        def patch(p,rot):
            im=Image.new("L",(80,80),0); d=ImageDraw.Draw(im)
            pts=[]
            for i in range(p*2):
                a=rot+i*math.pi/p; rr=32 if i%2==0 else 13
                pts.append((40+rr*math.cos(a),40+rr*math.sin(a)))
            d.polygon(pts,outline=255,width=3)
            return np.asarray(im,dtype=np.float32)/255.0
        a=shape_drag.radial_signature(patch(5,0.0))
        b=shape_drag.radial_signature(patch(5,1.1))
        c=shape_drag.radial_signature(patch(4,0.0))
        self.assertGreater(float(np.dot(a,b)), float(np.dot(a,c)))

    def test_garbage_input_returns_none(self):
        self.assertIsNone(shape_drag.solve_shape_drag(b""))
        self.assertIsNone(shape_drag.solve_shape_drag(b"notanimage"))

class TestRobustness(unittest.TestCase):
    """Faint, low-contrast glyphs must still yield candidates."""

    def _faint_scene(self):
        bg = Image.new("RGB", (400, 300), (128, 150, 160))
        bg = bg.filter(ImageFilter.GaussianBlur(6))
        d = ImageDraw.Draw(bg)
        def g(cx, cy, p, rot, col):
            pts = []
            for i in range(p * 2):
                a = rot + i * math.pi / p
                rr = 24 if i % 2 == 0 else 10
                pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
            d.polygon(pts, outline=col, width=2)
        # deliberately low contrast against the background
        for (x, y) in ((60, 110), (150, 80), (250, 120), (120, 200)):
            g(x, y, 5, 0.3, (150, 170, 178))
        d.rectangle((300, 40, 370, 110), fill=(210, 214, 216))
        g(335, 75, 5, 0.0, (160, 140, 175))
        b = io.BytesIO(); bg.save(b, format="PNG")
        return b.getvalue()

    def test_threshold_sweep_finds_faint_glyphs(self):
        got = shape_drag.solve_shape_drag(self._faint_scene())
        self.assertIsNotNone(got, "faint scene produced no pairing")
        self.assertEqual(got["type"], "drag")

    def test_logger_is_called(self):
        lines = []
        shape_drag.solve_shape_drag(self._faint_scene(),
                                    log=lambda m, **k: lines.append(m))
        self.assertTrue(lines, "solver must explain what it did")

    def test_never_raises_on_junk(self):
        for junk in (b"", b"xx", bytes(range(64))):
            self.assertIsNone(shape_drag.solve_shape_drag(junk))


if __name__=="__main__":
    unittest.main(verbosity=2)
