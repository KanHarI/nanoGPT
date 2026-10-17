# Example expressions by gamma

Ten procedural samples per evaluation gamma, with division and exponentiation disabled.
Operators: `+`, `-`, `*`. Seed: `20261017`; maximum operator count: 32.
A separate sampler is used, so examples do not affect training or evaluation randomness.

The prefix identifies the field (`b` = 11, `d` = 13, `h` = 17, `j` = 19).
Gamma controls the Poisson rate of the operator count; individual sizes vary.
All answers were checked with the independent stack evaluator.

## Gamma 0

```text
b.2=2
3.2=2
d.7=7
h.2=2
5.1=1
b.7=7
5.0=0
d.c=c
2.1=1
d.5=5
```

## Gamma 1

```text
j.ab-c+=b
d.2c+=1
5.22+2-=2
5.3=3
j.d=d
2.1=1
d.c=c
3.21+=0
5.4=4
j.c2cg-*+=4
```

## Gamma 4

```text
3.20+0*=0
3.0=0
j.bc-fi9--*=d
3.1=1
7.540+-=1
2.00-11-*0+0*=0
j.db*f-ff**=f
5.444+3*-2*=0
b.67*23+-=4
b.641-89*-*=4
```

## Gamma 8

```text
j.d7+7h--d9+5**7d*ad*--=e
d.66*06-7c86+*68+*--357*++-=c
3.122-010-*--021**-1-=0
3.10-20+-00--=2
7.310+*0-323-*-=6
5.21-3204*---033+**=0
2.100+*1-101+-00+0-*0*+=1
d.0a3+*3c+*=0
b.78-99-a-67*6*-0-6+*=3
5.01-3011+**2-*=2
```

## Gamma 16

```text
b.9a0+6-+27-62-*819*-61+85*-+**25+-=0
d.42-b6b-5-+*b0+*97*+114-4---09*4b-**=0
7.1121-16+*61++30-+*44***06250+-1++-*=4
h.fga--05**d66-*-0ag+d+-d1+**=0
2.01+001*01*+-01+-0+0*-10++10-1*10+*+=1
2.111-0+*100-+00-+100*011*-0*0++*0*-*00*1-+=1
5.31+3402**3+44***003-2*1+-42-*33++2***21*-=0
j.bb1-b*8-ii+h9-*-*993+*ag*3-14*--25-*+=8
3.12+0*010-+222*-2-0*-12+0-2-2*-2-*00-*001+1-++=0
h.013--geg*-6f-*7+g6+*597-+**=1
```
