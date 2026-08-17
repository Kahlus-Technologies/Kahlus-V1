# Say it at a table

No equations. This is the whole thing in words you can use out loud.
If someone pushes, the follow-ups are at the bottom.

---

## The 20-second version

> "We were trying to predict future brain activity from past brain activity. We got
> a result that looked incredible. Then we found out the model wasn't predicting
> anything — the answer was already inside the question. So we worked out exactly
> how much that mistake flatters you, and it turns out there's a clean formula for
> it. You can now look at somebody's paper and calculate how inflated their number
> is without touching their data."

## The 90-second version

**The setup.** EEG is brain activity recorded from electrodes on the scalp. We
were asking: given the last second or so of someone's brain signal, can a model
predict what comes next?

**The result that looked amazing.** Our model scored a correlation of 0.97.
Near-perfect. For brain data that's absurd — and absurd is a warning sign.

**What was actually happening.** We asked the model to predict a window of 127
numbers. But 126 of those 127 numbers were already sitting in the input we handed
it. It just copied them across. One number out of 127 was a genuine prediction,
and it got graded on all 127. It looked like forecasting. It was photocopying.

**Why nobody catches this.** The whole field guards against a different mistake —
the model secretly seeing the test data while training. There are careful
procedures for that, and we followed them, and they don't help. This mistake
isn't about training data leaking into test data. It's that *the answer is
inside the question*, in every single example. All the standard safeguards pass
and the number is still fake.

**What we did about it.** We worked out the exact formula for how much this
inflates your score. It's just the ratio of the window size to how far ahead
you're predicting. Predict 127 numbers where only 1 is genuinely new, and you
look up to 127 times better than you are. We checked it 286 different ways across
11 different kinds of signal. It never broke once.

**Why that's useful to other people.** You don't need their data or their code.
Those two numbers — window size and prediction distance — are printed in the
methods section of the paper. Divide them. That's the ceiling on how inflated
their result is.

**And then we did the experiment properly.** Scored honestly on 78 people, the
model does have real skill — but only about a sixth of a second into the future.
By a third of a second it's gone.

---

## If they push back

**"Isn't this just data leakage? That's well known."**
> "Different thing. Normal leakage is train data bleeding into test data — you fix
> it by splitting properly. We split properly. This is inside a *single example*:
> the thing you're asked to predict overlaps the thing you're given. Splitting
> can't fix it, because it isn't about the split."

**"How do you know the formula is right and not just fitted?"**
> "It's not fitted, it's derived. It falls out of the definition of the score.
> Then we checked it 286 times and never saw a violation. The best test was pure
> random noise — nothing predictable at all — and it still inflates by exactly
> the predicted amount. That proves it's about the *scoring rule*, not the data."

**"Did your own model actually turn out to be good?"**
> "Partly, and we say so both ways. Out to about 160 milliseconds it genuinely
> beats a properly-tuned classical baseline, on 26 of 27 people we held out. Past
> about 320 milliseconds the advantage is gone. We also had a second claim about
> long horizons that didn't survive our own checks, and we published that it
> failed rather than quietly dropping it."

**"Is 160 milliseconds useful for anything?"**
> "Not clinically, and I won't pretend otherwise. The thing we actually care about
> — predicting a state change before it happens — is on the order of a minute.
> That's hundreds of times further out. What we have right now is a methods
> result, not a medical one."

**"So what's the actual contribution?"**
> "Two things. A formula that lets anyone price how inflated a published
> forecasting number is, from the methods section alone. And a corrected benchmark
> that shows where a neural model really does beat a classical one and where it
> doesn't."

---

## Three things not to say

- **Don't say "we beat the state of the art."** We didn't, and someone already
  published a deep model beating classical baselines at this horizon (Pankka
  2024). Our contribution is the audit and the formula.
- **Don't say the 0.97 result was ours and real.** It was ours and wrong. Say that
  first — it lands better and it's the honest version.
- **Don't imply anything clinical.** No seizures, no consciousness, no diagnosis.

---

## The one-liner if you only get one sentence

> "We found that a common way of scoring brain-signal forecasts can make a model
> look up to a hundred times better than it is, we derived the exact formula for
> how much, and you can apply it to any published paper using two numbers from its
> methods section."
